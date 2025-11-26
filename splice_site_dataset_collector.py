from Bio import Entrez, SeqIO
import pandas as pd
import numpy as np
from collections import Counter
import re
import time
import os
import pickle
from datetime import datetime

# Configure Entrez
Entrez.email = "your.email@example.com"  # Replace with your email

class SpliceSiteDataCollector:
    #Handles data collection from NCBI GenBank/RefSeq

    def __init__(self, email, cache_dir="cache"):
        Entrez.email = email
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def search_human_genes(self, query="human[organism] AND biomol_mrna[PROP]", max_results=1000):
        #Search for human gene IDs from NCBI
        try:
            print(f"Searching NCBI for: {query}")
            handle = Entrez.esearch(
                db="nucleotide",
                term=query,
                retmax=max_results,
                usehistory="y"
            )
            results = Entrez.read(handle)
            handle.close()

            gene_ids = results["IdList"]
            print(f"Found {len(gene_ids)} gene IDs")
            return gene_ids
        except Exception as e:
            print(f"Error searching NCBI: {e}")
            return []

    def get_refseq_ids(self, max_results=500):
        # Get RefSeq human mRNA IDs
        query = 'human[organism] AND refseq[filter] AND biomol_mrna[PROP]'
        return self.search_human_genes(query, max_results)

    def fetch_sequences(self, gene_id, database="nucleotide"):
        # Fetch sequence data from NCBI with caching
        cache_file = os.path.join(self.cache_dir, f"{gene_id}.pkl")

        # Check cache first
        if os.path.exists(cache_file):
            print(f"Loading {gene_id} from cache")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Fetch from NCBI
        try:
            print(f"Fetching {gene_id} from NCBI")
            handle = Entrez.efetch(db=database, id=gene_id, rettype="gb", retmode="text")
            record = SeqIO.read(handle, "genbank")
            handle.close()

            # Cache the record
            with open(cache_file, 'wb') as f:
                pickle.dump(record, f)

            return record
        except Exception as e:
            print(f"Error fetching {gene_id}: {e}")
            return None

    def fetch_batch(self, gene_ids, max_genes=100, delay=0.34):
        #Fetch multiple gene sequences with rate limiting
        records = []

        for i, gene_id in enumerate(gene_ids[:max_genes]):
            print(f"\nProgress: {i+1}/{min(len(gene_ids), max_genes)}")
            record = self.fetch_sequences(gene_id)

            if record:
                records.append(record)

            # Rate limiting: 3 requests per second (NCBI limit)
            if i < len(gene_ids) - 1:
                time.sleep(delay)

        print(f"\nSuccessfully fetched {len(records)} records")
        return records

    def extract_splice_sites(self, record):
        #Extract annotated splice sites from GenBank record
        splice_sites = {
            'donor': [],
            'acceptor': []
        }

        # Method 1: Extract from multi-part CDS or mRNA features
        for feature in record.features:
            if feature.type in ["CDS", "mRNA"]:
                if hasattr(feature.location, 'parts') and len(feature.location.parts) > 1:
                    for i, part in enumerate(feature.location.parts):
                        if i < len(feature.location.parts) - 1:
                            # Donor site (end of exon)
                            splice_sites['donor'].append(int(part.end))
                        if i > 0:
                            # Acceptor site (start of exon)
                            splice_sites['acceptor'].append(int(part.start))

        # Method 2: If no multi-part features found, extract from exon boundaries
        if not splice_sites['donor'] and not splice_sites['acceptor']:
            exons = []
            for feature in record.features:
                if feature.type == "exon":
                    exons.append((int(feature.location.start), int(feature.location.end)))

            # Sort exons by start position
            exons.sort(key=lambda x: x[0])

            # Extract splice sites between consecutive exons
            for i in range(len(exons) - 1):
                # Donor site at end of current exon
                splice_sites['donor'].append(exons[i][1])
                # Acceptor site at start of next exon
                splice_sites['acceptor'].append(exons[i + 1][0])

        return splice_sites

    def build_dataset(self, gene_ids, preprocessor, max_genes=100, save_path="dataset.csv"):
        #Build complete dataset from multiple genes
        all_data = []
        stats = {
            'total_genes': 0,
            'successful_genes': 0,
            'total_donor_sites': 0,
            'total_acceptor_sites': 0,
            'total_samples': 0
        }

        print(f"\n{'='*60}")
        print(f"Building dataset from {min(len(gene_ids), max_genes)} genes")
        print(f"{'='*60}\n")

        for i, gene_id in enumerate(gene_ids[:max_genes]):
            stats['total_genes'] += 1
            print(f"\n[{i+1}/{min(len(gene_ids), max_genes)}] Processing {gene_id}")

            record = self.fetch_sequences(gene_id)

            if record:
                splice_sites = self.extract_splice_sites(record)

                if splice_sites['donor'] or splice_sites['acceptor']:
                    dataset = preprocessor.create_dataset(record.seq, splice_sites)
                    all_data.append(dataset)

                    stats['successful_genes'] += 1
                    stats['total_donor_sites'] += len(splice_sites['donor'])
                    stats['total_acceptor_sites'] += len(splice_sites['acceptor'])
                    stats['total_samples'] += len(dataset)

                    print(f"  ✓ Extracted {len(splice_sites['donor'])} donor, "
                          f"{len(splice_sites['acceptor'])} acceptor sites")
                    print(f"  ✓ Generated {len(dataset)} samples")
                else:
                    print(f"  ✗ No splice sites found")
            else:
                print(f"  ✗ Failed to fetch record")

            # Rate limiting
            if i < len(gene_ids) - 1:
                time.sleep(0.34)

        # Combine all datasets
        if all_data:
            final_dataset = pd.concat(all_data, ignore_index=True)
            final_dataset.to_csv(save_path, index=False)

            print(f"\n{'='*60}")
            print("Dataset Statistics:")
            print(f"{'='*60}")
            print(f"Genes processed: {stats['total_genes']}")
            print(f"Genes with splice sites: {stats['successful_genes']}")
            print(f"Total donor sites: {stats['total_donor_sites']}")
            print(f"Total acceptor sites: {stats['total_acceptor_sites']}")
            print(f"Total samples: {stats['total_samples']}")
            print(f"Positive samples: {final_dataset['label'].sum()}")
            print(f"Negative samples: {len(final_dataset) - final_dataset['label'].sum()}")
            print(f"\nDataset saved to: {save_path}")
            print(f"{'='*60}\n")

            return final_dataset
        else:
            print("\n✗ No data collected")
            return None


class SpliceSitePreprocessor:
    #Handles sequence extraction and preprocessing

    def __init__(self, flank_size=20):
        self.flank_size = flank_size

    def extract_flanking_sequence(self, sequence, position, site_type='donor'):
        #Extract flanking sequence around splice site
        start = max(0, position - self.flank_size)
        end = min(len(sequence), position + self.flank_size + 2)

        flanking_seq = str(sequence[start:end])
        return flanking_seq.upper()

    def generate_negative_samples(self, sequence, positive_sites, site_type='donor'):
        #Generate decoy GT/AG sequences that are not true splice sites
        negative_sites = []
        motif = 'GT' if site_type == 'donor' else 'AG'

        # Find all occurrences of the motif
        seq_str = str(sequence).upper()
        for match in re.finditer(motif, seq_str):
            pos = match.start()
            # Exclude positions that are actual splice sites
            if pos not in positive_sites:
                negative_sites.append(pos)

        return negative_sites

    def create_dataset(self, sequence, splice_sites):
        #Create balanced dataset with positive and negative examples\
        data = []

        # Process donor sites
        for pos in splice_sites['donor']:
            seq = self.extract_flanking_sequence(sequence, pos, 'donor')
            data.append({
                'sequence': seq,
                'position': pos,
                'type': 'donor',
                'label': 1
            })

        # Generate negative donor samples
        neg_donor = self.generate_negative_samples(
            sequence, splice_sites['donor'], 'donor'
        )
        for pos in neg_donor[:len(splice_sites['donor']) * 10]:  # 5:1 ratio
            seq = self.extract_flanking_sequence(sequence, pos, 'donor')
            data.append({
                'sequence': seq,
                'position': pos,
                'type': 'donor',
                'label': 0
            })

        # Process acceptor sites
        for pos in splice_sites['acceptor']:
            seq = self.extract_flanking_sequence(sequence, pos, 'acceptor')
            data.append({
                'sequence': seq,
                'position': pos,
                'type': 'acceptor',
                'label': 1
            })

        # Generate negative acceptor samples
        neg_acceptor = self.generate_negative_samples(
            sequence, splice_sites['acceptor'], 'acceptor'
        )
        for pos in neg_acceptor[:len(splice_sites['acceptor']) * 10]: #10:1 ratio
            seq = self.extract_flanking_sequence(sequence, pos, 'acceptor')
            data.append({
                'sequence': seq,
                'position': pos,
                'type': 'acceptor',
                'label': 0
            })

        return pd.DataFrame(data)


class FeatureEngineer:
    #Extracts features from sequences for ML models

    def __init__(self, k=3):
        self.k = k

    def get_kmer_frequencies(self, sequence, k=None):
        #Calculate k-mer frequencies
        if k is None:
            k = self.k

        kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        kmer_counts = Counter(kmers)
        total = sum(kmer_counts.values())

        return {kmer: count/total for kmer, count in kmer_counts.items()}

    def get_position_specific_composition(self, sequence):
        #Get nucleotide composition at each position
        features = {}
        for i, nucleotide in enumerate(sequence):
            features[f'pos_{i}_{nucleotide}'] = 1
        return features

    def get_gc_content(self, sequence):
        #Calculate GC content
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0

    def get_purine_pyrimidine_ratio(self, sequence):
        #Calculate purine to pyrimidine ratio
        purines = sequence.count('A') + sequence.count('G')
        pyrimidines = sequence.count('C') + sequence.count('T')
        return purines / pyrimidines if pyrimidines > 0 else 0

    def extract_all_features(self, sequence):
        #Extract all features for a sequence
        features = {}

        # K-mer frequencies
        kmer_freq = self.get_kmer_frequencies(sequence)
        features.update({f'kmer_{k}': v for k, v in kmer_freq.items()})

        # Position-specific composition
        pos_comp = self.get_position_specific_composition(sequence)
        features.update(pos_comp)

        # Sequence features
        features['gc_content'] = self.get_gc_content(sequence)
        features['purine_pyrimidine_ratio'] = self.get_purine_pyrimidine_ratio(sequence)
        features['length'] = len(sequence)

        return features

    def create_feature_matrix(self, df):
        #Create feature matrix from sequence dataframe
        feature_list = []

        for idx, row in df.iterrows():
            features = self.extract_all_features(row['sequence'])
            features['label'] = row['label']
            features['type'] = row['type']
            feature_list.append(features)

        feature_df = pd.DataFrame(feature_list)
        return feature_df.fillna(0)


# Example usage
if __name__ == "__main__":
    print("Splice Site Prediction Tool - Boilerplate")
    print("=" * 50)

    # Initialize components
    collector = SpliceSiteDataCollector("your.email@example.com")
    preprocessor = SpliceSitePreprocessor(flank_size=20)
    feature_engineer = FeatureEngineer(k=3)

    print("\nComponents initialized:")
    print("- Data Collector (with caching)")
    print("- Preprocessor")
    print("- Feature Engineer")

    # Example workflow options:
    print("\n" + "="*50)
    print("USAGE EXAMPLES:")
    print("="*50)

    # Option 1: Get RefSeq gene IDs and build dataset
    print("\n1. Build dataset from RefSeq genes:")
    print("   gene_ids = collector.get_refseq_ids(max_results=100)")
    print("   dataset = collector.build_dataset(gene_ids, preprocessor, max_genes=50)")
    print("   features = feature_engineer.create_feature_matrix(dataset)")

    # Option 2: Search for specific genes
    print("\n2. Search for specific genes:")
    print("   gene_ids = collector.search_human_genes('BRCA1[Gene Name] AND human[Organism]')")
    print("   dataset = collector.build_dataset(gene_ids, preprocessor)")

    # Option 3: Use specific gene IDs
    print("\n3. Use specific gene IDs:")
    print("   gene_ids = ['NM_000492', 'NM_000518', 'NM_000546']  # CFTR, HBB, TP53")
    print("   dataset = collector.build_dataset(gene_ids, preprocessor)")

    # Option 4: Load existing dataset
    print("\n4. Load existing dataset:")
    print("   dataset = pd.read_csv('dataset.csv')")
    print("   features = feature_engineer.create_feature_matrix(dataset)")

    print("\n" + "="*50)
    print("\nUncomment the code below to start building your dataset!")
    print("="*50 + "\n")

    # Save collected data
    print("\nFetching gene IDs from NCBI RefSeq...")
    gene_ids = collector.get_refseq_ids(max_results=100)
    #
    if gene_ids:
        print(f"\nBuilding dataset from {len(gene_ids)} genes...")
        dataset = collector.build_dataset(
            gene_ids,
            preprocessor,
            max_genes=50,  # Start with 50 genes
            save_path="splice_site_dataset.csv"
        )
    #
        if dataset is not None:
            print("\nExtracting features...")
            features = feature_engineer.create_feature_matrix(dataset)
            features.to_csv("splice_site_features.csv", index=False)
            print(f"Feature matrix shape: {features.shape}")
            print(f"Features saved to: splice_site_features.csv")
            print("\n✓ Dataset ready for ML training!")