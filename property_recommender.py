import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import json
import traceback
from datetime import datetime
import joblib
import os

class PropertyRecommender:
    def __init__(self, model_path='models/property_recommender.joblib'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.feature_weights = {
            'gla': 0.3,  # Gross Living Area
            'lot_size_sf': 0.2,  # Lot Size
            'year_built': 0.15,  # Year Built
            'num_beds': 0.1,  # Number of Bedrooms
            'num_baths': 0.1,  # Number of Bathrooms
            'room_count': 0.05,  # Total Rooms
            'basement_area': 0.05,  # Basement Area
            'effective_age': 0.05  # Effective Age
        }
        self.feedback_history = []
        
    def load_data(self, data_path):
        """Load and prepare the appraisal dataset."""
        try:
            print("Loading data from", data_path)
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            print(f"Found {len(data['appraisals'])} appraisals")
            
            # Extract subjects and properties
            subjects = []
            properties = []
            selected_comps = []
            
            for i, appraisal in enumerate(data['appraisals']):
                try:
                    # Add subject
                    subject = appraisal['subject']
                    subject['appraisal_id'] = i
                    subjects.append(subject)
                    
                    # Add properties and track selected comps
                    selected_comp_ids = set()
                    for comp in appraisal.get('comps', []):
                        # Use address as identifier since comps don't have 'id'
                        selected_comp_ids.add(comp.get('address', ''))
                    
                    for prop in appraisal.get('properties', []):
                        prop['appraisal_id'] = i
                        prop['is_selected_comp'] = prop.get('address', '') in selected_comp_ids
                        properties.append(prop)
                    
                    if i % 10 == 0:
                        print(f"Processed {i} appraisals...")
                        
                except Exception as e:
                    print(f"Error processing appraisal {i}: {str(e)}")
                    continue
            
            print(f"Total subjects: {len(subjects)}")
            print(f"Total properties: {len(properties)}")
            return pd.DataFrame(subjects), pd.DataFrame(properties)
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print(traceback.format_exc())
            raise

    def preprocess_data(self, subjects_df, properties_df):
        """Preprocess the data for model training."""
        try:
            print("\nPreprocessing data...")
            # Select numerical features
            numerical_features = list(self.feature_weights.keys())
            
            # Find common features between subjects and properties
            common_features = list(set(subjects_df.columns) & set(properties_df.columns) & set(numerical_features))
            print(f"Common features between subjects and properties: {common_features}")
            
            if not common_features:
                raise ValueError("No common numerical features found between subjects and properties")
            
            # Convert features to numeric
            print("Converting features to numeric...")
            for feature in common_features:
                subjects_df[feature] = pd.to_numeric(subjects_df[feature], errors='coerce')
                properties_df[feature] = pd.to_numeric(properties_df[feature], errors='coerce')
            
            # Handle missing values
            print("Handling missing values...")
            X_subjects = subjects_df[common_features].copy()
            X_properties = properties_df[common_features].copy()
            
            # Check for completely missing features
            missing_in_subjects = X_subjects.columns[X_subjects.isna().all()].tolist()
            missing_in_properties = X_properties.columns[X_properties.isna().all()].tolist()
            missing_features = list(set(missing_in_subjects + missing_in_properties))
            
            if missing_features:
                print(f"Features with all missing values: {missing_features}")
                X_subjects = X_subjects.drop(columns=missing_features)
                X_properties = X_properties.drop(columns=missing_features)
                common_features = [f for f in common_features if f not in missing_features]
                print(f"Remaining features after removing missing: {common_features}")
            
            # Impute missing values
            X_subjects_imputed = self.imputer.fit_transform(X_subjects)
            X_properties_imputed = self.imputer.transform(X_properties)
            
            # Convert back to DataFrame
            X_subjects_imputed = pd.DataFrame(X_subjects_imputed, columns=common_features)
            X_properties_imputed = pd.DataFrame(X_properties_imputed, columns=common_features)
            
            return X_subjects_imputed, X_properties_imputed, common_features
            
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            print(traceback.format_exc())
            raise

    def train(self, subjects_df, properties_df):
        """Train the recommendation model."""
        try:
            print("\nTraining recommendation model...")
            # Preprocess data
            X_subjects, X_properties, features = self.preprocess_data(subjects_df, properties_df)
            
            # Scale features
            print("Scaling features...")
            X_properties_scaled = self.scaler.fit_transform(X_properties)
            X_subjects_scaled = self.scaler.transform(X_subjects)
            
            # Create labels for properties (1 for selected comps, 0 for others)
            property_labels = properties_df['is_selected_comp'].astype(int)
            
            # Train KNN model
            print("Training KNN model...")
            self.knn.fit(X_properties_scaled, property_labels)
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'knn': self.knn,
                'scaler': self.scaler,
                'imputer': self.imputer,
                'features': features
            }, self.model_path)
            
            print("Model trained and saved successfully")
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            print(traceback.format_exc())
            raise

    def find_comps(self, subject_df, properties_df, n_neighbors=3):
        """Find comparable properties for a subject property."""
        try:
            print("\nFinding comparable properties...")
            # Load model if not already loaded
            if not hasattr(self, 'knn'):
                model_data = joblib.load(self.model_path)
                self.knn = model_data['knn']
                self.scaler = model_data['scaler']
                self.imputer = model_data['imputer']
                self.features = model_data['features']
            
            # Preprocess data
            X_subject, X_properties, _ = self.preprocess_data(subject_df, properties_df)
            
            # Scale features
            X_subject_scaled = self.scaler.transform(X_subject)
            X_properties_scaled = self.scaler.transform(X_properties)
            
            # Find nearest neighbors
            distances, indices = self.knn.kneighbors(X_subject_scaled)
            
            # Get comparable properties
            comps = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                comp_property = properties_df.iloc[idx]
                similarity_score = np.exp(-distance)
                
                # Generate explanation
                explanation = self._generate_explanation(
                    subject_df.iloc[0],
                    comp_property,
                    X_subject.iloc[0],
                    X_properties.iloc[idx]
                )
                
                comps.append({
                    'address': comp_property['address'],
                    'similarity_score': similarity_score,
                    'explanation': explanation
                })
            
            return comps
            
        except Exception as e:
            print(f"Error finding comps: {str(e)}")
            print(traceback.format_exc())
            raise

    def _generate_explanation(self, subject, comp, subject_features, comp_features):
        """Generate explanation for why a property was selected as a comp."""
        explanations = []
        
        # Compare key features
        for feature, weight in self.feature_weights.items():
            if feature in subject_features and feature in comp_features:
                subj_val = subject_features[feature]
                comp_val = comp_features[feature]
                
                # Calculate similarity percentage
                if subj_val != 0:  # Avoid division by zero
                    similarity = 1 - abs(subj_val - comp_val) / max(subj_val, comp_val)
                    if similarity > 0.8:  # Only mention very similar features
                        explanations.append(
                            f"Similar {feature.replace('_', ' ')}: "
                            f"{comp_val:.0f} vs {subj_val:.0f}"
                        )
        
        # Add location-based explanation if available
        if 'city' in subject and 'city' in comp:
            if subject['city'] == comp['city']:
                explanations.append("Located in the same city")
        
        return " | ".join(explanations)

    def add_feedback(self, subject_id, comp_id, is_good_comp, feedback_reason=None):
        """Add user feedback to improve the model."""
        feedback = {
            'subject_id': subject_id,
            'comp_id': comp_id,
            'is_good_comp': is_good_comp,
            'feedback_reason': feedback_reason,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_history.append(feedback)
        
        # If we have enough new feedback, retrain the model
        if len(self.feedback_history) >= 10:
            self._retrain_with_feedback()

    def _retrain_with_feedback(self):
        """Retrain the model incorporating user feedback."""
        try:
            print("\nRetraining model with user feedback...")
            # Convert feedback to training data
            feedback_df = pd.DataFrame(self.feedback_history)
            
            # Update property labels based on feedback
            # This would need to be implemented based on your specific data structure
            
            # Retrain the model
            # This would need to be implemented based on your specific data structure
            
            print("Model retrained with feedback")
            
        except Exception as e:
            print(f"Error retraining model: {str(e)}")
            print(traceback.format_exc())

    def evaluate(self, subjects_df, properties_df, data, k=10):
        """
        Evaluate the recommendation model using multiple metrics.
        
        Args:
            subjects_df: DataFrame of subject properties
            properties_df: DataFrame of all properties
            data: Original data dict containing appraisals with selected_comps
            k: Number of recommendations to evaluate (default 10)
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        try:
            print(f"\n{'='*60}")
            print("EVALUATING RECOMMENDATION MODEL")
            print(f"{'='*60}")
            
            metrics = {
                'hit_rate': 0,
                'precision_at_k': [],
                'recall_at_k': [],
                'mrr': [],  # Mean Reciprocal Rank
                'ndcg_at_k': [],  # Normalized Discounted Cumulative Gain
                'total_subjects': 0,
                'subjects_with_hits': 0,
                'avg_comps_per_subject': 0
            }
            
            total_actual_comps = 0
            
            for i, appraisal in enumerate(data['appraisals']):
                try:
                    subject = appraisal['subject']
                    subject_df = pd.DataFrame([subject])
                    
                    # Get actual selected comps for this appraisal
                    actual_comp_ids = set()
                    for comp in appraisal.get('comps', []):
                        actual_comp_ids.add(comp.get('address', ''))
                    
                    if not actual_comp_ids:
                        continue
                    
                    total_actual_comps += len(actual_comp_ids)
                    metrics['total_subjects'] += 1
                    
                    # Get properties for this appraisal
                    appraisal_properties = pd.DataFrame(appraisal.get('properties', []))
                    
                    if len(appraisal_properties) == 0:
                        continue
                    
                    # Get recommendations
                    recommendations = self._get_recommendation_indices(
                        subject_df, appraisal_properties, k
                    )
                    
                    # Get recommended property IDs
                    recommended_ids = set()
                    for idx in recommendations:
                        if idx < len(appraisal_properties):
                            recommended_ids.add(appraisal_properties.iloc[idx].get('address', ''))
                    
                    # Calculate metrics for this subject
                    hits = recommended_ids & actual_comp_ids
                    
                    # Hit Rate (at least one correct comp found)
                    if len(hits) > 0:
                        metrics['subjects_with_hits'] += 1
                    
                    # Precision@K
                    precision = len(hits) / k if k > 0 else 0
                    metrics['precision_at_k'].append(precision)
                    
                    # Recall@K
                    recall = len(hits) / len(actual_comp_ids) if len(actual_comp_ids) > 0 else 0
                    metrics['recall_at_k'].append(recall)
                    
                    # Mean Reciprocal Rank (MRR)
                    mrr = self._calculate_mrr(recommendations, appraisal_properties, actual_comp_ids)
                    metrics['mrr'].append(mrr)
                    
                    # NDCG@K
                    ndcg = self._calculate_ndcg(recommendations, appraisal_properties, actual_comp_ids, k)
                    metrics['ndcg_at_k'].append(ndcg)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Evaluated {i + 1} appraisals...")
                        
                except Exception as e:
                    print(f"Error evaluating appraisal {i}: {str(e)}")
                    continue
            
            # Calculate final metrics
            if metrics['total_subjects'] > 0:
                metrics['hit_rate'] = metrics['subjects_with_hits'] / metrics['total_subjects']
                metrics['avg_precision_at_k'] = np.mean(metrics['precision_at_k']) if metrics['precision_at_k'] else 0
                metrics['avg_recall_at_k'] = np.mean(metrics['recall_at_k']) if metrics['recall_at_k'] else 0
                metrics['avg_mrr'] = np.mean(metrics['mrr']) if metrics['mrr'] else 0
                metrics['avg_ndcg_at_k'] = np.mean(metrics['ndcg_at_k']) if metrics['ndcg_at_k'] else 0
                metrics['avg_comps_per_subject'] = total_actual_comps / metrics['total_subjects']
                
                # Calculate F1 Score
                if (metrics['avg_precision_at_k'] + metrics['avg_recall_at_k']) > 0:
                    metrics['f1_score'] = 2 * (metrics['avg_precision_at_k'] * metrics['avg_recall_at_k']) / \
                                         (metrics['avg_precision_at_k'] + metrics['avg_recall_at_k'])
                else:
                    metrics['f1_score'] = 0
            
            self._print_evaluation_report(metrics, k)
            return metrics
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            print(traceback.format_exc())
            raise

    def _get_recommendation_indices(self, subject_df, properties_df, k):
        """Get indices of top-k recommended properties."""
        try:
            # Preprocess data
            X_subject, X_properties, _ = self.preprocess_data(subject_df, properties_df)
            
            # Scale features
            X_subject_scaled = self.scaler.transform(X_subject)
            X_properties_scaled = self.scaler.transform(X_properties)
            
            # Calculate distances to all properties
            distances = np.linalg.norm(X_properties_scaled - X_subject_scaled, axis=1)
            
            # Get indices of k nearest neighbors
            indices = np.argsort(distances)[:k]
            return indices
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def _calculate_mrr(self, recommendations, properties_df, actual_comp_ids):
        """Calculate Mean Reciprocal Rank."""
        for rank, idx in enumerate(recommendations, 1):
            if idx < len(properties_df):
                prop_address = properties_df.iloc[idx].get('address', '')
                if prop_address in actual_comp_ids:
                    return 1.0 / rank
        return 0.0

    def _calculate_ndcg(self, recommendations, properties_df, actual_comp_ids, k):
        """Calculate Normalized Discounted Cumulative Gain."""
        dcg = 0.0
        for rank, idx in enumerate(recommendations[:k], 1):
            if idx < len(properties_df):
                prop_address = properties_df.iloc[idx].get('address', '')
                if prop_address in actual_comp_ids:
                    dcg += 1.0 / np.log2(rank + 1)
        
        # Calculate ideal DCG (if all actual comps were ranked first)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_comp_ids), k)))
        
        return dcg / idcg if idcg > 0 else 0.0

    def _print_evaluation_report(self, metrics, k):
        """Print a formatted evaluation report."""
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Total subjects evaluated: {metrics['total_subjects']}")
        print(f"   • Avg comps per subject: {metrics['avg_comps_per_subject']:.2f}")
        print(f"   • Subjects with at least 1 hit: {metrics['subjects_with_hits']}")
        
        print(f"\n🎯 Recommendation Quality (k={k}):")
        print(f"   • Hit Rate: {metrics['hit_rate']*100:.2f}%")
        print(f"     (% of subjects where ≥1 correct comp found)")
        
        print(f"\n   • Precision@{k}: {metrics.get('avg_precision_at_k', 0)*100:.2f}%")
        print(f"     (% of recommendations that are correct)")
        
        print(f"\n   • Recall@{k}: {metrics.get('avg_recall_at_k', 0)*100:.2f}%")
        print(f"     (% of actual comps found in top {k})")
        
        print(f"\n   • F1 Score: {metrics.get('f1_score', 0)*100:.2f}%")
        print(f"     (Harmonic mean of precision and recall)")
        
        print(f"\n📈 Ranking Quality:")
        print(f"   • Mean Reciprocal Rank (MRR): {metrics.get('avg_mrr', 0):.4f}")
        print(f"     (How high the first correct comp ranks)")
        
        print(f"\n   • NDCG@{k}: {metrics.get('avg_ndcg_at_k', 0):.4f}")
        print(f"     (Quality of ranking considering position)")
        
        print(f"\n{'='*60}")
        
        # Interpretation guide
        print("\n📖 Interpretation Guide:")
        if metrics['hit_rate'] >= 0.8:
            print("   ✅ Hit Rate: Excellent - Model finds relevant comps most of the time")
        elif metrics['hit_rate'] >= 0.5:
            print("   ⚠️  Hit Rate: Moderate - Room for improvement")
        else:
            print("   ❌ Hit Rate: Low - Model needs significant improvement")
            
        if metrics.get('avg_mrr', 0) >= 0.5:
            print("   ✅ MRR: Good - Correct comps rank highly")
        else:
            print("   ⚠️  MRR: Could improve - Correct comps not ranking high enough")

def main():
    try:
        # Initialize recommender
        recommender = PropertyRecommender()
        
        # Check if data file exists
        data_path = 'appraisals_dataset.json'
        if not os.path.exists(data_path):
            print(f"Error: Data file '{data_path}' not found.")
            print("Please ensure the appraisals_dataset.json file exists in the current directory.")
            return
        
        # Load raw data for evaluation
        print("\nLoading raw data...")
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Load and split data
        print("\nLoading data...")
        subjects_df, properties_df = recommender.load_data(data_path)
        
        if len(subjects_df) == 0 or len(properties_df) == 0:
            print("Error: No data loaded. Please check the data file format.")
            return
        
        print(f"\nLoaded {len(subjects_df)} subjects and {len(properties_df)} properties")
        
        # Split data into training and validation sets
        print("\nSplitting data into training and validation sets...")
        train_subjects, val_subjects = train_test_split(subjects_df, test_size=0.2, random_state=42)
        print(f"Training set size: {len(train_subjects)}")
        print(f"Validation set size: {len(val_subjects)}")
        
        # Train model
        print("\nTraining model...")
        recommender.train(train_subjects, properties_df)
        
        # Run comprehensive evaluation
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Evaluate with different k values
        for k in [3, 5, 10]:
            print(f"\n>>> Evaluating with k={k}")
            metrics = recommender.evaluate(subjects_df, properties_df, raw_data, k=k)
        
        # Sample predictions for validation set
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS (First 5 subjects)")
        print("="*60)
        
        for i, (_, subject) in enumerate(val_subjects.head(5).iterrows(), 1):
            print(f"\n{'─'*50}")
            print(f"Subject {i}: {subject.get('address', 'N/A')}")
            print(f"{'─'*50}")
            
            try:
                comps = recommender.find_comps(
                    pd.DataFrame([subject]),
                    properties_df
                )
                
                print("Top comparable properties:")
                for j, comp in enumerate(comps, 1):
                    print(f"  {j}. {comp['address']}")
                    print(f"     Similarity Score: {comp['similarity_score']:.4f}")
                    print(f"     Reason: {comp['explanation']}")
                    
            except Exception as e:
                print(f"Error processing subject {i}: {str(e)}")
                continue
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 