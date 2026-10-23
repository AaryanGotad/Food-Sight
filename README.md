![FoodSight Cover](assets/food_sight_cover.png)

# FoodSight

FoodSight is an image-classification streamlit application that answers a simple question: **what dish is in this image?** Upload a food photograph, choose one of the bundled examples, and FoodSight uses a fine-tuned EfficientNet-B0 model to rank the image against the 101 categories in the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

The app is built with Streamlit and TensorFlow. It is designed to make the prediction process inspectable rather than presenting a single unexplained label: the interface shows the resized image supplied to the model, the leading prediction, a confidence state, and a top-five probability breakdown.

---

## Live Demo
The entire application is deployed on Streanlit Community Cloud, [check it out here!](https://jhx9yh6q73obk3cqkbsev6.streamlit.app/)

---

## Features

- Classifies images into 101 Food-101 dish categories.
- Accepts PNG, JPG, and JPEG uploads.
- Includes one local sample image for every supported class.
- Provides quick sample buttons for popular dishes such as pizza, sushi, ramen, and tacos.
- Resizes every input to the model's required 224 x 224 RGB tensor.
- Displays the model's top prediction and top-five outputs.
- Labels results as `CONFIDENT`, `UNCERTAIN`, or `CONFUSED` using the prediction probability and the gap between the first two predictions.
- Shows the resized model input beside the original image.
- Uses Streamlit resource and data caching so the model and class list are not reloaded unnecessarily.
- Protects inference with a lock, which keeps model access serialized if multiple Streamlit interactions reach prediction at once.

---

## Live Workflow

The prediction flow is:

```mermaid
flowchart TD
    A[Upload an image or choose a sample] --> B[Convert image to RGB]
    B --> C[Resize to 224 x 224]
    C --> D[Cast to float32 and add batch dimension]
    D --> E[FoodSight.keras]
    E --> F[101-way softmax output]
    F --> G[Extract top five classes]
    G --> H{Prediction state}
    H -->|Top probability >= 0.35 and margin >= 0.15| I[CONFIDENT]
    H -->|Top probability < 0.35| J[CONFUSED]
    H -->|Otherwise| K[UNCERTAIN]
```

### 1. Choose an image

There are two ways to provide an image:

- Upload a PNG, JPG, or JPEG through the file uploader.
- Select a dish from the searchable list or popular sample pills, then load its bundled sample image.

The repository contains 101 sample JPGs under `assets/samples/`, matching the 101 lines in `utils/class_names.txt`.

<img src="assets/docImages/choose_img.png" style="width: 50%;">

### 2. Prepare the image

Before inference, `utils.utilities.preprocess_image()`:

1. Converts non-RGB images, including grayscale images and images with transparency, to RGB.
2. Converts the PIL image into a TensorFlow tensor.
3. Resizes it to 224 x 224 pixels.
4. Casts the pixel values to `float32`.
5. Adds a batch dimension, producing a tensor with shape `(1, 224, 224, 3)`.

The app displays a visual 224 x 224 preview so it is clear what spatial representation reaches the classifier. The preview is a resized image representation; the actual tensor is the float32 tensor returned by the preprocessing function.

### 3. Generate predictions

`model/FoodSight.keras` returns 101 values, one for each class in `utils/class_names.txt`. The class index is important: the output at index `i` is interpreted using the class name at line `i` in that file.

The app selects the five largest values with TensorFlow's `tf.math.top_k`. The table exposes both:

- **Confidence %**: the same value formatted as a percentage with two decimal places.
- **Confidence Score**: the underlying probability-like value in the interval from 0 to 1.

<img src="assets/docImages/pred1.png" style="width: 50%;">
<br>
<img src="assets/docImages/pred2.png" style="width: 50%;">

---

## Understanding the Confidence State

![confusion state cover](assets/docImages/confusion_state_cover.png)

FoodSight does not treat every top prediction as equally persuasive. After extracting the top results, it calculates:

```text
margin = probability of first prediction - probability of second prediction
```

The current rules are:

| State | Rule | Meaning |
|---|---|---|
| `CONFIDENT` | Top probability is at least `0.35` and the margin is at least `0.15` | One class is sufficiently strong and clearly ahead of the runner-up. |
| `CONFUSED` | Top probability is below `0.35` | No class has a strong enough score to support a clear result. |
| `UNCERTAIN` | Any other case | The leading class may be plausible, but the runner-up is close enough to show both. |

<img src="assets/docImages/conf_pred1.png" style="width: 50%;">
<br>
<img src="assets/docImages/conf_pred2.png" style="width: 50%;">

These are application display rules, **not** a guarantee that the model is correct. The model was trained to choose among a fixed set of 101 dishes. An image of an unsupported dish, a poorly framed image, or a visually ambiguous dish will **still** receive a best guess because the classifier has no explicit `unknown food` output.

---

## Why Can Softmax Show `1.0` and Tiny Other Values?

The saved model's final layer is a TensorFlow softmax activation. In exact mathematics, softmax is:

<img src="assets/docImages/softmax.png" style="margin-left: 15%; width: 50%;">

where `z` is the vector of raw model scores, or logits. The ideal probabilities are non-negative and sum to one.

For a very confident prediction, the winning logit can be much larger than the others. After the usual numerically stable subtraction of the largest logit, the winning term is approximately `1` and the remaining exponential terms can be extremely small, for example `10^-4` or `10^-5`.

The model runs with finite-precision floating-point arithmetic rather than exact real numbers. When the denominator is accumulated, contributions that are small compared with the precision of the current value can be rounded away. The winning output can therefore be stored as exactly `1.0`, while the losing outputs remain representable as small positive numbers. Adding the stored values may then produce a result slightly greater than one.

That observation does **not** mean the model has produced a mathematically valid probability distribution with a total greater than one. It means the computed float32 values contain a small numerical normalization error. It is especially visible for overconfident predictions because the gap between the winning logit and the other logits is large.

There are two additional display details worth keeping in mind:

- The interface shows only the top five of 101 outputs. Their displayed values are not expected to add up to 100% because the other 96 classes are omitted.
- Percentages are rounded to two decimal places, so the displayed table can differ slightly from the underlying values used by the progress bars and state logic.

### How to investigate it

The current app exposes the raw top-five scores in the **Raw Model Outputs** popover. For a deeper diagnosis, inspect the complete model output before selecting the top five and compare:

```python
outputs = model.predict(processed_image, verbose=0)[0]
print(outputs.dtype)
print(outputs.sum())
print(outputs.max())
```

The saved model has a softmax output, so the sum should be very close to `1.0`; a tiny deviation is expected from floating-point computation.

---

## Model and Dataset

FoodSight uses a fine-tuned EfficientNet-B0 classifier trained on the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). The project was motivated by the goal of exceeding [DeepFood](https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment)'s reported 77.4% top-1 accuracy while gaining experience with a larger image-classification dataset.

### Dataset

![dataset cover](assets/docImages/dataset_cover.png)

Food-101 contains 101 food categories. The training split contains 75,750 images, with 750 images per class. The official validation split contains 25,250 images, with 250 images per class. For my experiments, 15% of that validation split, or 3,788 images, was used for validation during training. The downloaded dataset occupied approximately 4.65 GiB, sourced from `TensorFlow Datasets`.

Every image was resized to 224 x 224 pixels and cast from `uint8` to `float32`. Training data was shuffled with a buffer size of 1,000, batched in groups of 32, and prefetched with TensorFlow's `AUTOTUNE`. Validation data was batched and prefetched without shuffling to keep experiment result comparisons consistent.

The model predicts the following 101 categories, among others:

`apple_pie`, `baklava`, `bibimbap`, `cannoli`, `chicken_curry`, `chocolate_cake`, `french_fries`, `hamburger`, `ice_cream`, `lasagna`, `pizza`, `ramen`, `sushi`, `tacos`, `tiramisu`, `waffles`, and many others.

The complete class order is stored in [`utils/class_names.txt`](utils/class_names.txt). It must remain aligned with the model's output indices. Renaming, reordering, or deleting entries will make otherwise valid predictions display the wrong dish name.

The model file is [`model/FoodSight.keras`](model/FoodSight.keras). A runtime inspection of the saved artifact confirms:

| Property | Value |
|---|---|
| Input shape | `(None, 224, 224, 3)` |
| Output shape | `(None, 101)` |
| Final activation | Softmax |
| Runtime | TensorFlow / Keras |

### Training Setup

Training used TensorFlow mixed precision with the `mixed_float16` policy where compatible hardware was available. The EfficientNet-B0 base model was loaded without its classification head and initially frozen. A new classification head was added with:

- Random horizontal flips, rotations, translations, and contrast augmentation.
- Global average pooling.
- Dropout with a rate of `0.3`.
- A dense layer with 101 outputs.
- A float32 softmax activation for stable output probabilities.

The model was compiled with `sparse categorical cross-entropy` loss and the `Adam` optimizer. The first feature-extraction stage ran for 5 epochs while the base model remained frozen. Its validation result was `72.78%` accuracy with a loss of `0.9974`.

The second stage unfroze the base model for fine-tuning, with a maximum of 100 epochs. Training stopped after 26 epochs. The run used model checkpoints, early stopping after 3 validation-loss evaluations without improvement, and `ReduceLROnPlateau` with a factor of `0.2`, patience of 2, and a minimum learning rate of `1e-7`.

### Fine-Tuning Results

The reported results were measured on the 15% validation subset used during training, rather than on the complete Food-101 validation split:

| Metric | Score |
|---|---:|
| Accuracy | 82.68% |
| Precision | 82.59% |
| Recall | 82.68% |
| F1 score | 82.59% |
| Loss | 0.6266 |

The fine-tuned model therefore exceeded the 77.4% top-1 accuracy reported by [DeepFood](https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment) by 5.28 percentage points on this experiment's validation subset. These results should not be treated as a directly controlled reproduction of the DeepFood evaluation protocol.

### Confusion Matrix

The confusion matrix below shows the class-level behavior of the trained model on the evaluation used for this project. The axes follow the same class order as [`utils/class_names.txt`](utils/class_names.txt).

![FoodSight confusion matrix](assets/docImages/confusion_matrix.png)

**Key Observations:**

*   **Strong Diagonal Performance:** The prominent, solid dark blue diagonal line from the top-left to the bottom-right visually confirms that the model correctly predicts the true label across the vast majority of the 101 classes. This aligns with the model's high 82.68% validation accuracy.
*   **Normalized Evaluation:** The matrix is normalized on a 0.0 to 1.0 scale, representing the percentage of images correctly classified per class. The consistently dark diagonal indicates that the model's accuracy is well-distributed and balanced, rather than relying heavily on a few dominant classes.
*   **Sparse Misclassifications:** The pale, light blue off-diagonal spots indicate occasional misclassifications. Because there are no heavily darkened off-diagonal clusters, the EfficientNet-B0 model does not exhibit systemic failure modes or extreme bias toward guessing a single incorrect dish.
*   **Expected Visual Ambiguity:** The scattered off-diagonal noise reflects the inherent difficulty of the Food-101 dataset, where dishes with similar plating, textures, and colors (e.g., chocolate cake versus chocolate mousse) naturally share visual features.

---

## Repository Structure

```text
Food-Sight/
├── app.py                    # Streamlit user interface and inference flow
├── model/
│   └── FoodSight.keras       # Saved TensorFlow/Keras classifier
├── utils/
│   ├── __init__.py
│   ├── class_names.txt       # 101 class names in model-output order
│   └── utilities.py          # Preprocessing, top-k extraction, state logic
├── assets/
│   ├── docImages/            # Documentation images, including the confusion matrix
│   ├── samples/              # One JPG sample for each supported class
│   └── silc_readme.md        # README from the earlier SILC project
├── requirements.txt          # Runtime dependencies
├── pyproject.toml            # Project metadata and dependency constraints
├── model prob reaching.txt   # Notes on softmax floating-point behavior
├── LICENSE
└── README.md
```

---

## Local Setup

### Requirements

- Python 3.11 or newer.
- A CPU capable of running TensorFlow inference.
- Enough disk space for the TensorFlow runtime and the saved model.

### Install

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The dependency set includes Streamlit, Pillow, TensorFlow CPU, and NumPy. The project metadata in `pyproject.toml` also requires Python `>=3.11`.

### Run

```bash
.venv/bin/streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Open [http://localhost:8501](http://localhost:8501) in a browser. On first load, Streamlit loads and caches the Keras model. CPU-only inference can take a little longer on the first prediction while TensorFlow initializes.

---

## API and Inference Notes

FoodSight is currently a Streamlit application, not a separate REST API. Predictions are made inside the Streamlit process when the user presses **Identify This**.

The core inference call is conceptually:

```python
processed_image = utilities.preprocess_image(image)
predictions = model.predict(processed_image)
top_predictions = utilities.top_k_preds(predictions)
```

`top_k_preds()` returns five dictionaries with a `label` and a raw `confidence` value. The UI then keeps the raw value as `Probability` and converts a copy to a percentage for presentation.

---

## Limitations

- **Fixed label set:** The classifier can only choose among 101 Food-101 categories. It cannot reliably identify a dish outside those categories.
- **Best-guess behavior:** Softmax always distributes scores across the known classes; a high score is not proof that the image belongs to the dataset.
- **Visual ambiguity:** Dishes with similar colors, textures, or plating can be difficult to distinguish, such as chocolate cake versus chocolate mousse.
- **Framing and image quality:** Heavy cropping, poor lighting, packaging, clutter, or a dish occupying only a small part of the image can reduce performance.
- **Resize information loss:** Every image is resized to 224 x 224, so very small details may disappear or become distorted.
- **Confidence calibration:** The displayed score is the model's softmax output, not a formally calibrated probability of correctness.

---

## Good Input Practices

To give FoodSight best possible chances to see what's on your plate:

- Keep the dish as the main subject and place it near the center.
- Use clear, well-lit images with enough visible surface area.
- Avoid strong obstructions such as hands, wrappers, or utensils covering most of the food.
- Choose an angle that exposes the dish's characteristic shape and texture.
- Treat `CONFUSED` and `UNCERTAIN` results as prompts to inspect the top-five alternatives, not as definitive identifications.

---

## Acknowledgement

FoodSight was made possible by the following open-source projects, datasets, and resources:

- The [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) by Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool, which provides the 101 food categories and images used for training and evaluation.
- [TensorFlow Datasets](https://www.tensorflow.org/datasets), used to obtain and prepare the Food-101 data during the training workflow.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/), which provide the model-training and inference frameworks.
- [EfficientNet](https://arxiv.org/abs/1905.11946), specifically EfficientNet-B0, which provides the image-classification backbone used by FoodSight.
- [Streamlit](https://streamlit.io/), which powers the interactive web application and its deployment workflow.
- [Pillow](https://python-pillow.org/) and [NumPy](https://numpy.org/), which support image loading and input preparation.
- [DeepFood](https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment), whose reported results provided a reference point for the project.

---

## License

See [`LICENSE`](LICENSE) for the project's license terms.
