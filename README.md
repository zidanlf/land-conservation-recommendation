# Land Conservation Recommendation System with CatBoost

Sistem rekomendasi konservasi lahan berbasis machine learning menggunakan algoritma CatBoost untuk multi-label classification. Model dapat memprediksi teknik konservasi yang tepat dan memberikan rekomendasi tanaman berdasarkan karakteristik lahan seperti fungsi kawasan, kemiringan lereng, jenis tanah, dan faktor lingkungan lainnya.

## 🚀 Features

- Multi-label classification menggunakan CatBoost
- Rekomendasi teknik konservasi lahan yang spesifik
- Sistem rekomendasi tanaman terintegrasi berdasarkan metode konservasi
- Handling missing values dengan preprocessing otomatis
- Support untuk data kategorik dan numerik
- GPU acceleration untuk training yang lebih cepat
- Model persistence dengan format .cbm dan .pkl
- JSON input support untuk integrasi aplikasi
- Comprehensive evaluation metrics

## 📋 Dataset Features

### Input Features
1. **FungsiKaw**: Fungsi kawasan (Hutan Produksi, Hutan Lindung, APL, dll.)
2. **chbulan**: Curah hujan bulanan
3. **Lereng**: Kemiringan lereng (Landai, Curam, Sangat Curam)
4. **Solumtnh**: Kedalaman solum tanah
5. **JnsLahan**: Jenis lahan (Lahan Kering, Lahan Basah)
6. **LC2024**: Land cover/penggunaan lahan tahun 2024
7. **ErosiPot**: Potensi erosi (Ringan, Sedang, Berat)
8. **TBE**: Tingkat Bahaya Erosi

### Output Labels (Multi-label)
- 42+ teknik konservasi lahan yang berbeda
- Meliputi: Agroforestry, Alley cropping, Contouring, Cover crop, dll.
- Setiap prediksi disertai confidence score

## 🛠️ Requirements

### System Requirements
- Python 3.7+
- GPU support (CUDA) untuk training optimal
- Minimum 8GB RAM
- Google Colab compatible

### Python Libraries
```bash
pip install catboost pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## 📁 Project Structure

```
land-conservation-system/
│
├── catboost_konservasi_lahan.py    # Main training script
├── catboost_predict.py             # Prediction script
├── EP_19mei2024_dissolved.xlsx     # Training dataset
├── catboost_multilabel_model.cbm   # Trained model file
├── model_components.pkl             # Model components
├── requirements.txt                 # Python dependencies
└── README.md                       # Project documentation
```

## 🚀 Usage

### 1. Model Training

```python
# Run training script
python catboost_konservasi_lahan.py
```

**Training Process:**
- Loads and preprocesses Excel dataset
- Creates multi-label binary encoding
- Handles missing values automatically
- Trains CatBoost multi-label classifier
- Evaluates model performance
- Saves trained model and components

### 2. Making Predictions

```python
# Import prediction module
from catboost_predict import predict_conservation_and_plants, json_to_dataframe

# JSON input format
json_input = {
    "data": [
        {
            "FungsiKaw": "Areal Penggunaan Lain (APL)",
            "chbulan": None,
            "Lereng": "Curam", 
            "Solumtnh": None,
            "JnsLahan": "Lahan Kering",
            "LC2024": "Kebun Campur",
            "ErosiPot": "Ringan",
            "TBE": "Sedang"
        }
    ]
}

# Convert to DataFrame and predict
new_data = json_to_dataframe(json_input)
recommendations, plants = predict_conservation_and_plants(new_data)
```

### 3. Standalone Prediction

```python
python catboost_predict.py
```

## 📊 Model Architecture

### CatBoost Configuration
- **Loss Function**: MultiLogloss (multi-label classification)
- **Iterations**: 2000 with early stopping
- **Learning Rate**: 0.05
- **Depth**: 8
- **Task Type**: GPU accelerated
- **Evaluation Metric**: MultiLogloss

### Feature Engineering
- **Categorical Encoding**: Native CatBoost handling
- **Missing Values**: Automatic preprocessing with "UNKNOWN" placeholder
- **Multi-label Binarization**: Using sklearn's MultiLabelBinarizer

## 📈 Model Performance

### Evaluation Metrics
- **Macro F1-Score**: Multi-class average performance
- **Micro F1-Score**: Overall classification performance  
- **Hamming Loss**: Multi-label prediction accuracy
- **Classification Report**: Per-class precision, recall, F1-score
- **Feature Importance**: Identifies most influential factors

### Key Features by Importance
1. Land cover (LC2024)
2. Slope gradient (Lereng)
3. Erosion potential (ErosiPot)
4. Soil depth (Solumtnh)
5. Forest function (FungsiKaw)

## 🌱 Plant Recommendation System

### Integrated Plant Database
- **100+ plant species** mapped to conservation techniques
- **Categories**: MPTS (Multi-Purpose Tree Species), NTFP (Non-Timber Forest Products), Commercial timber
- **Characteristics**: High evapotranspiration, deep rooting systems, water storage capacity

### Example Plant Recommendations
- **Agroforestry MPTS**: Alpukat, Durian, Mangga, Kemiri
- **Commercial Timber**: Meranti, Mahang, Sengon, Tusam
- **NTFP**: Karet, Rotan, Damar, Cengkeh

## 🔧 API Integration

### JSON Input Format
```json
{
    "data": [
        {
            "FungsiKaw": "string",
            "chbulan": "string or null",
            "Lereng": "string", 
            "Solumtnh": "string or null",
            "JnsLahan": "string",
            "LC2024": "string",
            "ErosiPot": "string",
            "TBE": "string"
        }
    ]
}
```

### Response Format
```python
recommendations = [
    ("Conservation Method", confidence_score, ["Plant1", "Plant2"]),
    # ... more recommendations
]
all_plants = ["Plant1", "Plant2", "Plant3", ...]
```

## 📊 Data Preprocessing

### Missing Value Handling
```python
def preprocess_data_for_catboost(df, categorical_cols, missing_value_placeholder="UNKNOWN"):
    # Replaces NaN in categorical columns with "UNKNOWN"
    # CatBoost handles NaN in numerical columns naturally
    return processed_df
```

### Multi-label Processing
```python
# Convert comma-separated labels to binary matrix
df['Rekomendasi_List'] = df['Rekomendasi_Konservasi'].apply(lambda x: [item.strip() for item in x.split(', ')])
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(df['Rekomendasi_List'])
```

## 🎯 Use Cases

### 1. Agricultural Planning
- Soil conservation strategy selection
- Crop rotation planning with conservation techniques
- Sustainable farming practice recommendations

### 2. Forest Management
- Reforestation species selection
- Watershed protection planning
- Biodiversity conservation strategies

### 3. Land Rehabilitation
- Post-mining land restoration
- Degraded land recovery programs
- Erosion control implementation

### 4. Environmental Consulting
- EIA (Environmental Impact Assessment) support
- Land use planning recommendations
- Climate adaptation strategies

## 🚀 Deployment Options

### 1. Google Colab (Current)
```python
# Upload files and run directly in Colab
from google.colab import files
files.upload()  # Upload your data
```

### 2. Local Deployment
```bash
git clone [repository-url]
cd land-conservation-system
pip install -r requirements.txt
python catboost_predict.py
```

### 3. Web API Integration
- Flask/FastAPI wrapper untuk REST API
- JSON input/output untuk web applications
- Model serving dengan cloud platforms

## 🔍 Model Validation

### Cross-validation Strategy
- Stratified split berdasarkan label distribution
- 80-20 train-test split dengan random_state=42
- Early stopping untuk mencegah overfitting

### Performance Monitoring
- Feature importance analysis
- Prediction confidence thresholding
- Multi-label evaluation metrics


## 🙏 Acknowledgments

- Dataset: EP_19mei2024_dissolved.xlsx (Land conservation planning data, private dataset)
- CatBoost team for the excellent gradient boosting framework
- Scikit-learn community for preprocessing tools

---

**Note**: Model accuracy depends on data quality. Ensure proper field validation before implementing conservation recommendations in real-world scenarios.
