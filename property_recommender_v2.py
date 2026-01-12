import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import json
import traceback
from datetime import datetime
import joblib
import os
import re
from difflib import SequenceMatcher

class PropertyRecommenderV2:
    """
    Redesigned Property Recommender with:
    1. Better feature extraction and parsing
    2. Fuzzy address matching for evaluation
    3. Weighted multi-feature similarity scoring
    4. Geographic proximity consideration
    5. Robust handling of missing data
    """
    
    def __init__(self, model_path='models/property_recommender_v2.joblib'):
        self.model_path = model_path
        self.scaler = MinMaxScaler()  # 0-1 scaling for interpretable weights
        self.imputer = SimpleImputer(strategy='median')
        
        # Optimized feature weights based on appraisal importance
        self.feature_weights = {
            'gla': 0.25,           # Gross Living Area - most important
            'bedrooms': 0.15,      # Number of Bedrooms
            'room_count': 0.10,    # Total Room Count
            'year_built': 0.15,    # Age/Year Built
            'lot_size_sf': 0.10,   # Lot Size
            'full_baths': 0.10,    # Bathrooms
            'close_price': 0.15,   # Sale Price (important for comps)
        }
        
        self.features_used = []
        self.trained = False
        
    def _parse_numeric(self, value):
        """Extract numeric value from string with units."""
        if pd.isna(value) or value is None:
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        
        # Convert to string and clean
        val_str = str(value).lower().strip()
        
        # Handle common patterns
        if val_str in ['n/a', 'na', 'none', '', 'unknown']:
            return np.nan
        
        # Remove commas and extract number
        val_str = val_str.replace(',', '')
        
        # Handle sqft/sqm patterns like "1044 SqFt" or "1500.67 sqft +/-"
        match = re.search(r'([\d.]+)', val_str)
        if match:
            num = float(match.group(1))
            # Convert sqm to sqft if needed
            if 'sqm' in val_str:
                num *= 10.7639
            return num
        
        return np.nan
    
    def _parse_year(self, value):
        """Parse year built from various formats."""
        if pd.isna(value) or value is None:
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        
        val_str = str(value).strip()
        match = re.search(r'(\d{4})', val_str)
        if match:
            return float(match.group(1))
        return np.nan
    
    def _parse_price(self, value):
        """Parse price from various formats."""
        if pd.isna(value) or value is None:
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        
        val_str = str(value).replace(',', '').replace('$', '').strip()
        match = re.search(r'([\d.]+)', val_str)
        if match:
            return float(match.group(1))
        return np.nan
    
    def _normalize_address(self, address):
        """Normalize address for fuzzy matching."""
        if pd.isna(address) or address is None:
            return ""
        
        addr = str(address).lower().strip()
        
        # Remove common suffixes and standardize
        replacements = [
            (r'\s+', ' '),           # Multiple spaces to single
            (r'street', 'st'),
            (r'avenue', 'ave'),
            (r'drive', 'dr'),
            (r'road', 'rd'),
            (r'boulevard', 'blvd'),
            (r'place', 'pl'),
            (r'court', 'ct'),
            (r'crescent', 'cres'),
            (r'circle', 'cir'),
            (r'lane', 'ln'),
            (r'north', 'n'),
            (r'south', 's'),
            (r'east', 'e'),
            (r'west', 'w'),
            (r'[,.]', ''),           # Remove punctuation
            (r'\s+(on|ab|bc|sk|mb|qc|ns|nb|pe|nl|nt|nu|yt)\s*', ' '),  # Province codes
            (r'\s+[a-z]\d[a-z]\s*\d[a-z]\d\s*$', ''),  # Postal codes
        ]
        
        for pattern, replacement in replacements:
            addr = re.sub(pattern, replacement, addr)
        
        return addr.strip()
    
    def _address_similarity(self, addr1, addr2):
        """Calculate similarity between two addresses."""
        norm1 = self._normalize_address(addr1)
        norm2 = self._normalize_address(addr2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def load_data(self, data_path):
        """Load and prepare the appraisal dataset with enhanced parsing."""
        try:
            print("Loading data from", data_path)
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            print(f"Found {len(data['appraisals'])} appraisals")
            
            subjects = []
            properties = []
            
            for i, appraisal in enumerate(data['appraisals']):
                try:
                    # Process subject
                    subject = appraisal['subject'].copy()
                    subject['appraisal_id'] = i
                    
                    # Parse subject features
                    subject['gla_parsed'] = self._parse_numeric(subject.get('gla'))
                    subject['lot_size_parsed'] = self._parse_numeric(subject.get('lot_size_sf'))
                    subject['year_built_parsed'] = self._parse_year(subject.get('year_built'))
                    subject['bedrooms_parsed'] = self._parse_numeric(subject.get('num_beds'))
                    subject['room_count_parsed'] = self._parse_numeric(subject.get('room_count'))
                    
                    subjects.append(subject)
                    
                    # Get comp addresses for matching
                    comp_addresses = set()
                    for comp in appraisal.get('comps', []):
                        comp_addresses.add(self._normalize_address(comp.get('address', '')))
                    
                    # Process properties
                    for prop in appraisal.get('properties', []):
                        prop_copy = prop.copy()
                        prop_copy['appraisal_id'] = i
                        
                        # Check if this property matches any comp (fuzzy)
                        prop_addr_norm = self._normalize_address(prop.get('address', ''))
                        is_comp = False
                        for comp_addr in comp_addresses:
                            if self._address_similarity(prop_addr_norm, comp_addr) > 0.7:
                                is_comp = True
                                break
                        
                        prop_copy['is_selected_comp'] = is_comp
                        properties.append(prop_copy)
                    
                    if i % 20 == 0:
                        print(f"Processed {i} appraisals...")
                        
                except Exception as e:
                    print(f"Error processing appraisal {i}: {str(e)}")
                    continue
            
            subjects_df = pd.DataFrame(subjects)
            properties_df = pd.DataFrame(properties)
            
            # Count selected comps
            n_comps = properties_df['is_selected_comp'].sum()
            print(f"\nTotal subjects: {len(subjects_df)}")
            print(f"Total properties: {len(properties_df)}")
            print(f"Matched comps in properties: {n_comps}")
            
            return subjects_df, properties_df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
            raise

    def preprocess_features(self, properties_df):
        """Extract and preprocess features from properties."""
        
        features_df = pd.DataFrame()
        
        # GLA (Gross Living Area)
        features_df['gla'] = properties_df['gla'].apply(self._parse_numeric)
        
        # Bedrooms
        features_df['bedrooms'] = properties_df['bedrooms'].apply(self._parse_numeric)
        
        # Room count
        features_df['room_count'] = properties_df['room_count'].apply(self._parse_numeric)
        
        # Year built
        features_df['year_built'] = properties_df['year_built'].apply(self._parse_year)
        
        # Lot size
        features_df['lot_size_sf'] = properties_df['lot_size_sf'].apply(self._parse_numeric)
        
        # Bathrooms
        features_df['full_baths'] = properties_df['full_baths'].apply(self._parse_numeric)
        
        # Close price
        features_df['close_price'] = properties_df['close_price'].apply(self._parse_price)
        
        return features_df

    def train(self, subjects_df, properties_df):
        """Train the recommendation model using weighted feature similarity."""
        try:
            print("\n" + "="*60)
            print("TRAINING RECOMMENDATION MODEL V2")
            print("="*60)
            
            # Preprocess features
            print("\nExtracting features...")
            features_df = self.preprocess_features(properties_df)
            
            # Determine which features have enough data
            feature_coverage = {}
            for col in features_df.columns:
                coverage = features_df[col].notna().mean()
                feature_coverage[col] = coverage
                print(f"  {col}: {coverage*100:.1f}% coverage")
            
            # Use features with >30% coverage
            self.features_used = [f for f, cov in feature_coverage.items() if cov > 0.3]
            print(f"\nFeatures selected for model: {self.features_used}")
            
            if not self.features_used:
                raise ValueError("No features with sufficient coverage!")
            
            # Prepare feature matrix
            X = features_df[self.features_used].copy()
            
            # Impute missing values with median
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=self.features_used
            )
            
            # Scale features to 0-1
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_imputed),
                columns=self.features_used
            )
            
            # Store scaled features for later use
            self.property_features = X_scaled
            self.property_ids = properties_df.index.tolist()
            
            # Build nearest neighbors index for fast retrieval
            self.nn_model = NearestNeighbors(
                n_neighbors=min(50, len(X_scaled)),
                metric='euclidean',
                algorithm='ball_tree'
            )
            self.nn_model.fit(X_scaled)
            
            self.trained = True
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else '.', exist_ok=True)
            joblib.dump({
                'scaler': self.scaler,
                'imputer': self.imputer,
                'features_used': self.features_used,
                'nn_model': self.nn_model,
                'property_features': self.property_features,
            }, self.model_path)
            
            print("\n✅ Model trained successfully!")
            print(f"   Features: {self.features_used}")
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            traceback.print_exc()
            raise

    def _compute_weighted_similarity(self, subject_features, property_features):
        """Compute weighted similarity between subject and properties."""
        
        # Get weights for features we're using
        weights = np.array([
            self.feature_weights.get(f, 0.1) for f in self.features_used
        ])
        weights = weights / weights.sum()  # Normalize
        
        # Compute weighted Euclidean distance
        diff = property_features - subject_features.values
        weighted_diff = diff * weights
        distances = np.sqrt((weighted_diff ** 2).sum(axis=1))
        
        # Convert to similarity (higher is better)
        similarities = 1 / (1 + distances)
        
        return similarities

    def find_comps(self, subject_df, properties_df, n_neighbors=10):
        """Find comparable properties for a subject."""
        try:
            if not self.trained:
                raise ValueError("Model not trained! Call train() first.")
            
            # Preprocess subject features
            subject_features = pd.DataFrame()
            subject = subject_df.iloc[0]
            
            for feature in self.features_used:
                if feature == 'gla':
                    val = self._parse_numeric(subject.get('gla', subject.get('gla_parsed')))
                elif feature == 'bedrooms':
                    val = self._parse_numeric(subject.get('num_beds', subject.get('bedrooms')))
                elif feature == 'room_count':
                    val = self._parse_numeric(subject.get('room_count'))
                elif feature == 'year_built':
                    val = self._parse_year(subject.get('year_built'))
                elif feature == 'lot_size_sf':
                    val = self._parse_numeric(subject.get('lot_size_sf'))
                elif feature == 'full_baths':
                    # Parse num_baths like "1:1" (full:half)
                    baths = subject.get('num_baths', '')
                    if isinstance(baths, str) and ':' in baths:
                        val = float(baths.split(':')[0])
                    else:
                        val = self._parse_numeric(baths)
                elif feature == 'close_price':
                    val = np.nan  # Subject doesn't have close price
                else:
                    val = self._parse_numeric(subject.get(feature))
                
                subject_features[feature] = [val]
            
            # Impute and scale
            subject_imputed = pd.DataFrame(
                self.imputer.transform(subject_features),
                columns=self.features_used
            )
            subject_scaled = pd.DataFrame(
                self.scaler.transform(subject_imputed),
                columns=self.features_used
            )
            
            # Preprocess properties for this appraisal
            prop_features = self.preprocess_features(properties_df)
            prop_imputed = pd.DataFrame(
                self.imputer.transform(prop_features[self.features_used]),
                columns=self.features_used
            )
            prop_scaled = pd.DataFrame(
                self.scaler.transform(prop_imputed),
                columns=self.features_used
            )
            
            # Compute similarities
            similarities = self._compute_weighted_similarity(subject_scaled.iloc[0], prop_scaled)
            
            # Get top N
            top_indices = np.argsort(similarities)[::-1][:n_neighbors]
            
            comps = []
            for idx in top_indices:
                prop = properties_df.iloc[idx]
                similarity = similarities[idx]
                
                # Generate explanation
                explanation = self._generate_explanation(subject, prop, subject_features.iloc[0], prop_features.iloc[idx])
                
                comps.append({
                    'address': prop.get('address', 'N/A'),
                    'similarity_score': float(similarity),
                    'explanation': explanation,
                    'gla': prop.get('gla'),
                    'bedrooms': prop.get('bedrooms'),
                    'year_built': prop.get('year_built'),
                    'close_price': prop.get('close_price'),
                })
            
            return comps
            
        except Exception as e:
            print(f"Error finding comps: {str(e)}")
            traceback.print_exc()
            return []

    def _generate_explanation(self, subject, comp, subject_features, comp_features):
        """Generate explanation for why a property was selected."""
        explanations = []
        
        feature_names = {
            'gla': 'GLA',
            'bedrooms': 'Beds',
            'room_count': 'Rooms',
            'year_built': 'Year',
            'lot_size_sf': 'Lot',
            'full_baths': 'Baths',
            'close_price': 'Price'
        }
        
        for feature in self.features_used:
            subj_val = subject_features.get(feature)
            comp_val = comp_features.get(feature) if hasattr(comp_features, 'get') else comp_features[feature]
            
            if pd.notna(subj_val) and pd.notna(comp_val) and subj_val != 0:
                diff_pct = abs(subj_val - comp_val) / max(abs(subj_val), 1) * 100
                if diff_pct < 15:  # Within 15%
                    name = feature_names.get(feature, feature)
                    if feature == 'close_price':
                        explanations.append(f"{name}: ${comp_val:,.0f}")
                    elif feature == 'year_built':
                        explanations.append(f"{name}: {int(comp_val)}")
                    else:
                        explanations.append(f"{name}: {comp_val:.0f}")
        
        return " | ".join(explanations) if explanations else "Similar property characteristics"

    def evaluate(self, subjects_df, properties_df, data, k=10):
        """Evaluate the model with comprehensive metrics."""
        try:
            print(f"\n{'='*60}")
            print(f"EVALUATING MODEL (k={k})")
            print(f"{'='*60}")
            
            metrics = {
                'total_subjects': 0,
                'subjects_with_hits': 0,
                'precision_at_k': [],
                'recall_at_k': [],
                'mrr': [],
                'ndcg_at_k': [],
            }
            
            total_actual_comps = 0
            
            for i, appraisal in enumerate(data['appraisals']):
                try:
                    subject = appraisal['subject']
                    subject_df = pd.DataFrame([subject])
                    
                    # Get actual comp addresses (normalized)
                    actual_comp_addrs = set()
                    for comp in appraisal.get('comps', []):
                        actual_comp_addrs.add(self._normalize_address(comp.get('address', '')))
                    
                    if not actual_comp_addrs:
                        continue
                    
                    metrics['total_subjects'] += 1
                    total_actual_comps += len(actual_comp_addrs)
                    
                    # Get properties for this appraisal
                    appraisal_properties = pd.DataFrame(appraisal.get('properties', []))
                    if len(appraisal_properties) == 0:
                        continue
                    
                    # Get recommendations
                    recommendations = self.find_comps(subject_df, appraisal_properties, n_neighbors=k)
                    
                    # Match recommended addresses to actual comps
                    hits = 0
                    first_hit_rank = None
                    dcg = 0.0
                    
                    for rank, rec in enumerate(recommendations, 1):
                        rec_addr = self._normalize_address(rec['address'])
                        
                        # Check if this recommendation matches any actual comp
                        is_match = False
                        for actual_addr in actual_comp_addrs:
                            if self._address_similarity(rec_addr, actual_addr) > 0.7:
                                is_match = True
                                break
                        
                        if is_match:
                            hits += 1
                            if first_hit_rank is None:
                                first_hit_rank = rank
                            dcg += 1.0 / np.log2(rank + 1)
                    
                    # Calculate metrics
                    if hits > 0:
                        metrics['subjects_with_hits'] += 1
                    
                    metrics['precision_at_k'].append(hits / k)
                    metrics['recall_at_k'].append(hits / len(actual_comp_addrs))
                    metrics['mrr'].append(1.0 / first_hit_rank if first_hit_rank else 0.0)
                    
                    # NDCG
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_comp_addrs), k)))
                    metrics['ndcg_at_k'].append(dcg / idcg if idcg > 0 else 0.0)
                    
                    if (i + 1) % 20 == 0:
                        print(f"  Evaluated {i + 1} appraisals...")
                        
                except Exception as e:
                    continue
            
            # Compute final metrics
            results = {
                'total_subjects': metrics['total_subjects'],
                'subjects_with_hits': metrics['subjects_with_hits'],
                'hit_rate': metrics['subjects_with_hits'] / metrics['total_subjects'] if metrics['total_subjects'] > 0 else 0,
                'avg_precision_at_k': np.mean(metrics['precision_at_k']) if metrics['precision_at_k'] else 0,
                'avg_recall_at_k': np.mean(metrics['recall_at_k']) if metrics['recall_at_k'] else 0,
                'avg_mrr': np.mean(metrics['mrr']) if metrics['mrr'] else 0,
                'avg_ndcg_at_k': np.mean(metrics['ndcg_at_k']) if metrics['ndcg_at_k'] else 0,
                'avg_comps_per_subject': total_actual_comps / metrics['total_subjects'] if metrics['total_subjects'] > 0 else 0,
            }
            
            # F1 Score
            if (results['avg_precision_at_k'] + results['avg_recall_at_k']) > 0:
                results['f1_score'] = 2 * (results['avg_precision_at_k'] * results['avg_recall_at_k']) / \
                                     (results['avg_precision_at_k'] + results['avg_recall_at_k'])
            else:
                results['f1_score'] = 0
            
            self._print_results(results, k)
            return results
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            traceback.print_exc()
            raise

    def _print_results(self, metrics, k):
        """Print formatted evaluation results."""
        print(f"\n{'─'*60}")
        print("📊 EVALUATION RESULTS")
        print(f"{'─'*60}")
        
        print(f"\n📈 Dataset:")
        print(f"   • Subjects evaluated: {metrics['total_subjects']}")
        print(f"   • Avg comps per subject: {metrics['avg_comps_per_subject']:.1f}")
        print(f"   • Subjects with ≥1 hit: {metrics['subjects_with_hits']}")
        
        print(f"\n🎯 Recommendation Quality (k={k}):")
        print(f"   • Hit Rate:      {metrics['hit_rate']*100:6.2f}%")
        print(f"   • Precision@{k}:  {metrics['avg_precision_at_k']*100:6.2f}%")
        print(f"   • Recall@{k}:     {metrics['avg_recall_at_k']*100:6.2f}%")
        print(f"   • F1 Score:      {metrics['f1_score']*100:6.2f}%")
        
        print(f"\n📊 Ranking Quality:")
        print(f"   • MRR:           {metrics['avg_mrr']:6.4f}")
        print(f"   • NDCG@{k}:       {metrics['avg_ndcg_at_k']:6.4f}")
        
        print(f"\n{'─'*60}")
        
        # Interpretation
        if metrics['hit_rate'] >= 0.7:
            print("✅ Hit Rate: Excellent!")
        elif metrics['hit_rate'] >= 0.4:
            print("⚠️  Hit Rate: Good, room for improvement")
        else:
            print("❌ Hit Rate: Needs improvement")


def main():
    try:
        print("\n" + "="*60)
        print("PROPERTY RECOMMENDER V2 - OPTIMIZED MODEL")
        print("="*60)
        
        recommender = PropertyRecommenderV2()
        
        data_path = 'appraisals_dataset.json'
        if not os.path.exists(data_path):
            print(f"Error: Data file '{data_path}' not found.")
            return
        
        # Load raw data
        print("\nLoading raw data...")
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Load and preprocess data
        subjects_df, properties_df = recommender.load_data(data_path)
        
        if len(subjects_df) == 0 or len(properties_df) == 0:
            print("Error: No data loaded.")
            return
        
        # Train model
        recommender.train(subjects_df, properties_df)
        
        # Evaluate with different k values
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE EVALUATION")
        print("="*60)
        
        all_results = {}
        for k in [3, 5, 10]:
            results = recommender.evaluate(subjects_df, properties_df, raw_data, k=k)
            all_results[k] = results
        
        # Summary comparison
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        print(f"\n{'k':<5} {'Hit Rate':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MRR':<10}")
        print("-" * 65)
        for k, r in all_results.items():
            print(f"{k:<5} {r['hit_rate']*100:>10.2f}% {r['avg_precision_at_k']*100:>10.2f}% "
                  f"{r['avg_recall_at_k']*100:>10.2f}% {r['f1_score']*100:>10.2f}% {r['avg_mrr']:>8.4f}")
        
        # Sample predictions
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS (First 3 subjects)")
        print("="*60)
        
        for i, appraisal in enumerate(raw_data['appraisals'][:3]):
            subject = appraisal['subject']
            subject_df = pd.DataFrame([subject])
            appraisal_props = pd.DataFrame(appraisal.get('properties', []))
            
            print(f"\n{'─'*50}")
            print(f"Subject: {subject.get('address', 'N/A')}")
            print(f"  GLA: {subject.get('gla')} | Beds: {subject.get('num_beds')} | Year: {subject.get('year_built')}")
            print(f"{'─'*50}")
            
            # Actual comps
            print("\nActual Comps (from appraisal):")
            for j, comp in enumerate(appraisal.get('comps', [])[:3], 1):
                print(f"  {j}. {comp.get('address')} - GLA: {comp.get('gla')}")
            
            # Predicted comps
            print("\nPredicted Comps:")
            comps = recommender.find_comps(subject_df, appraisal_props, n_neighbors=5)
            for j, comp in enumerate(comps[:3], 1):
                print(f"  {j}. {comp['address']}")
                print(f"     Score: {comp['similarity_score']:.4f} | {comp['explanation']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
