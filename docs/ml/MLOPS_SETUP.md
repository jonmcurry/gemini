# MLOps Setup and Lifecycle for Claims Processing Model

## Introduction

This document outlines the strategy and key considerations for Machine Learning Operations (MLOps) for the claims processing model used in this application. The goal is to establish a robust and repeatable process for training, deploying, and monitoring ML models to ensure continued performance and reliability.

A basic placeholder training script can be found at `models/training_scripts/train_model.py`. This script currently uses mock data for demonstration purposes and serves as an initial template. For a production-ready MLOps pipeline, this script and the surrounding processes will need to be significantly adapted to use real data and incorporate more advanced MLOps practices.

## 1. Data Sourcing and Preprocessing for Training

The foundation of any good ML model is high-quality, representative data.

*   **Data Source:** Training data should consist of historical claims, including all relevant features available at the time of processing and the corresponding outcomes (e.g., approval, rejection, identification of anomalies, as defined in `REQUIREMENTS.MD`). This data would typically be sourced from the production database (e.g., `claims_production` table after claims have been fully processed and outcomes are known) or a data warehouse.
*   **Data Cleaning & Transformation:** Raw data will likely require cleaning (handling missing values, correcting errors) and transformation (e.g., encoding categorical variables, scaling numerical features).
*   **Feature Engineering:** While the application's `FeatureExtractor` provides real-time feature extraction, training might benefit from more extensive offline feature engineering. This could involve creating new features from existing ones or incorporating additional data sources not available at inference time (though this can lead to train-serve skew if not managed carefully).
*   **Dataset Versioning:** Create versioned, static datasets for training, validation, and testing. This ensures reproducibility and allows for rollback or comparison across different model versions. These datasets should be stored reliably (e.g., S3, Google Cloud Storage, Azure Blob Storage, or a versioned database table).
*   **Tools:**
    *   **Data Manipulation:** Pandas, Dask (for larger datasets).
    *   **Preprocessing:** Scikit-learn.
    *   **Storage:** Cloud storage solutions, data lakes, or feature stores.

## 2. Model Training and Experimentation

This phase involves building and refining the ML model.

*   **Adapting `train_model.py`:** The existing script needs to be modified to load and preprocess real data from the versioned datasets.
*   **Experiment Tracking:** It is crucial to log all relevant information for each training run:
    *   Parameters used (e.g., learning rate, batch size, model architecture details).
    *   Code versions (Git commit hash).
    *   Performance metrics on training, validation, and test sets.
    *   Model artifacts (the trained model itself, visualizations like confusion matrices).
    *   **Tools:** MLflow, Weights & Biases, Kubeflow Pipelines (KFP) metadata, Vertex AI Experiments, SageMaker Experiments.
*   **Hyperparameter Tuning:** Employ automated techniques to find the optimal set of hyperparameters for the model.
    *   **Tools:** KerasTuner, Optuna, Ray Tune, Hyperopt.
*   **Model Architecture:** While the initial model is a simple Deep Neural Network (DNN), explore and evaluate other architectures if performance plateaus or if specific data characteristics suggest alternatives (e.g., Gradient Boosting Machines like XGBoost or LightGBM, especially if tabular data characteristics are dominant).

## 3. Model Optimization (TensorFlow Lite Conversion & Quantization)

For deployment in the application, the TensorFlow model needs to be converted to TensorFlow Lite (TFLite) format for efficiency.

*   **Conversion Process:**
    1.  Start with a trained TensorFlow SavedModel (as produced by `train_model.py`).
    2.  Use `tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)` to create a converter object.
*   **Optimization Strategies:**
    *   Apply standard optimizations: `converter.optimizations = [tf.lite.Optimize.DEFAULT]`. This typically includes techniques like constant folding and operator fusion, primarily targeting latency and size reduction.
*   **Quantization (8-bit Integer):**
    *   **Purpose:** Significantly reduces model size (up to 4x) and can improve inference speed, especially on hardware with specialized support for integer math.
    *   **Process:**
        1.  **Representative Dataset:** A small dataset (100-500 samples) that reflects the distribution of inputs the model will see in production is required for calibration during full integer quantization. This dataset should be passed to the converter.
        2.  **Dynamic Range Quantization:** Weights are quantized to 8-bit integers, activations are dynamically quantized at runtime. Simpler, no representative dataset needed, but less performance gain than full integer.
            `converter.optimizations = [tf.lite.Optimize.DEFAULT]` (already includes this usually)
        3.  **Full Integer Quantization:** All weights and activations are quantized to 8-bit integers. Requires a representative dataset.
            ```python
            def representative_dataset_gen():
                for i in range(num_calibration_samples):
                    # Yield a sample from your representative dataset
                    yield [input_features[i:i+1]]
            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8
            converter.inference_output_type = tf.int8 # or tf.uint8
            ```
        4.  **Quantization-Aware Training (QAT):** Simulates quantization effects during training, potentially yielding better accuracy for quantized models. More complex to implement.
    *   **Trade-offs:** Quantization can sometimes lead to a minor drop in model accuracy, which must be evaluated.
*   **Saving the TFLite Model:**
    ```python
    tflite_model = converter.convert()
    with open('models/claims_model.tflite', 'wb') as f:
        f.write(tflite_model)
    ```

## 4. Model Versioning

Proper model versioning is essential for reproducibility, rollback, and tracking.

*   **Strategies:**
    *   **Directory Structure:** Maintain numbered versions in a structured directory format, as started by `train_model.py` (e.g., `models/saved_model/claims_model/<version>/` for SavedModel, and a parallel structure like `models/tflite_models/claims_model/<version_number>.tflite` for converted models).
    *   **Git LFS (Large File Storage):** Store model binary files directly in Git, with Git LFS handling the large file pointers. This keeps model versions tied to code versions.
    *   **Dedicated Model Registry:** For more advanced lifecycle management (staging, production, archiving), use a dedicated model registry.
        *   **Tools:** MLflow Model Registry, Google Vertex AI Model Registry, AWS SageMaker Model Registry, Azure ML Model Registry.
*   **Association:** Crucially, each model version must be associated with the version of the data it was trained on, the version of the training code (Git commit), and its performance metrics.

## 5. Model Deployment

This refers to making the trained and optimized (`.tflite`) model available to the application.

*   **Loading Mechanism:** The `OptimizedPredictor` class in the application loads the TFLite model file.
*   **Updating the Model Path:**
    *   The active model is specified by the `ML_MODEL_PATH` setting in `settings.py`.
    *   **Manual Update:** For simple deployments, this path can be updated manually, followed by an application restart/redeploy.
    *   **CI/CD Automation:** A CI/CD pipeline can automate updating this configuration (e.g., in a Kubernetes ConfigMap or an environment variable) as part of a new model release process.
    *   **Dynamic Loading (Advanced):** `OptimizedPredictor` could be enhanced to poll a model registry or a designated path for new model versions and load them dynamically, potentially without a full application restart (requires careful handling of state and concurrency).
*   **Deployment Strategies:** If model updates are frequent or critical, consider:
    *   **Rolling Updates:** Gradually update application instances with the new model.
    *   **Blue-Green Deployments:** Deploy a new version of the application with the new model alongside the old one, then switch traffic once the new version is verified.

## 6. A/B Testing (Online Experimentation)

As per `REQUIREMENTS.MD`, A/B testing support is desired for comparing model versions in production.

*   **Concept:** Deploy multiple model versions (e.g., the current production model and a challenger) simultaneously. A portion of live traffic is routed to each version, and their performance is compared.
*   **Implementation Considerations:**
    *   **Routing/Traffic Splitting:**
        *   **API Gateway/Service Mesh:** Tools like Istio, Linkerd, or cloud-specific API gateways can manage traffic splitting.
        *   **Application-Level Logic:** The application itself (e.g., `ParallelClaimsProcessor` or an intermediary service) could decide which model version to call based on user/claim ID, a random percentage, or feature flags.
        *   **Feature Flagging Systems:** Tools like LaunchDarkly, Unleash, or Flagsmith can manage which users/requests are routed to which model.
    *   **Metrics Collection:** Ensure that predictions and relevant business outcomes are logged separately for each model version being tested. This allows for direct performance comparison (e.g., approval rates, detected anomalies, impact on processing times).
    *   **Analysis & Rollout:** Analyze the A/B test results to determine the winning model. Gradually roll out the winner to 100% of traffic.

## 7. Production Model Monitoring

Continuous monitoring of the deployed model is vital for detecting issues and degradation. This aligns with the "Model performance tracking and drift detection" requirement.

*   **Key Metrics to Monitor:**
    *   **Technical Performance:**
        *   Prediction Latency: Time taken for the model to make a prediction (covered by `ML_INFERENCE_DURATION_SECONDS` metric).
        *   Model Error Rate: Application-level errors from the model predictor (e.g., TFLite interpreter errors).
    *   **Model Output Performance:**
        *   Prediction Distribution: Monitor the distribution of `ml_score` (the model's output probability). Significant shifts can indicate issues.
        *   Decision Rates: Track the rates of "ML_APPROVED", "ML_REJECTED", and other `ml_derived_decision` categories.
    *   **Model Accuracy (Requires Ground Truth):**
        *   For claims processing, ground truth (e.g., actual fraud, correct payment) might be delayed. A system needs to be in place to eventually match model predictions with actual outcomes.
        *   Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC, Confusion Matrix.
*   **Drift Detection:**
    *   **Data Drift:** Monitor the statistical distribution of input features to the model in production. If these distributions change significantly from the data the model was trained on, model performance may degrade.
        *   **Tools/Techniques:** Statistical tests (e.g., Kolmogorov-Smirnov), population stability index (PSI).
    *   **Concept Drift:** Monitor the relationship between input features and the target variable. If this relationship changes over time (e.g., fraud patterns evolve), model performance will degrade even if input data distributions are stable. This is primarily detected by a drop in model accuracy metrics.
*   **Monitoring Tools:**
    *   **Prometheus/Grafana:** For technical metrics, some prediction distribution metrics.
    *   **Specialized ML Monitoring Platforms:** Tools like WhyLabs, Arize, Fiddler, Evidently AI offer more comprehensive drift detection, explainability, and bias monitoring.
    *   **Custom Dashboards:** BI tools or custom web applications for visualizing key performance indicators.
*   **Retraining Triggers:** Define criteria for when a model should be retrained. This could be:
    *   Significant performance degradation (e.g., F1-score drops below a threshold).
    *   Detection of significant data or concept drift.
    *   Scheduled retraining (e.g., quarterly) regardless of performance, to incorporate new data.

## Conclusion

Effective MLOps is an iterative process that requires collaboration between data scientists, ML engineers, and operations teams. This document provides a foundational outline. The specific tools and depth of implementation for each stage will evolve based on the project's needs, scale, and maturity.
```
