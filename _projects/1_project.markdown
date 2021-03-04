---
layout: distill
title: Learning Accurate ML Explainations with REAL-X and EVAL-X
description: How do we efficeintly generate ML explainations we can trust?
img: /assets/img/realx.png
importance: 1


authors:
  - name: Neil Jethani
    affiliations:
      name: NYU Langone Health, NYU
  - name: Mukund Sudarshan
    affiliations:
      name: NYU
  - name: Yin Aphinyanaphongs
    affiliations:
      name: NYU Langone Health
  - name: Rajesh Ranganath
    affiliations:
      name: NYU

---


## Why Do We Need Interpretability in Machine Learning?

Machine learning models are now pervasive throughout society, often making life or death decisions (self-driving cards, diagnosing lung cancer) or performing super-human tasks (mastering the game of Go). Given this not only do we need to trust machine learning models, but we should aim to learn from them. 

*For the rest of this post, I would like us to imagine that we are physcians.*  

Physcians need to be able to interpret why clinical decisions are made by ML models.
1. To trust treatment or diagnostic decisions made by models
2. To identify model failure modes so they can defer to their own judgement
3. To expand clinical knowledge

It's natural to then ask...

## What is an Explaination?

A physician may want to understand which genes are invovled in diabetes. However, often physicians want more personalized infromation about their patient. Instead, it would be more useful to understand which of their patient's genes are contributing to their diabetes so they can more effectively develope treatment plans. 

Accordingly, most interpretability methods focus on providing *local* explainations. *Local* or instance-wise explainations provide reasons for why a specfic decision was made. This is often accomplished by providing the feature importances related to that decision.

Now let's answer...

## What Do We Want in an Explaination?

As physicians, we need to be able to make quick, accurate decisions in order to treat patient's effectively. Therefore, we want explainations that are...
1. Fast
2. Accurate/ High Fidelity
3. Simple

Now that we know we want...

## Can We Have What We Want?

Let's consider existing interpretability methods and break them into three groups: 

|   | Gradient Based Methods    | Locally Linear Methods    | Perturbation Methods  |
|:-:    |:-:    |:-:    |:-:    |
| What?     | Measure the gradient of output   with respect to the input features?    | Uses a linear function of  simplified variables to explain  the prediction of a single input  | Perturb the inputs and observe  the effect on the target or  neurons within a network     |
| Examples  | gradCAM, DeepLift   | LIME, SHAP    | Occlusion     |
| Why (not)?    | Explanations don't optimize for accuracy/fidelity. Recent work shows estimates of  feature importance often do not  help identify features  that help predict the target anymore  than randomly assigned importances.     | This methods are slow, requiring numerous  perturbations of the input and/or  training a new model  to generate a single explanation.     | These methods are slow,  requiring numerous perturbations  to generate a single explanation.      |

Of note, both locally linear and perturbation-based methods rely on removing or perturbing features in order to characterize how/if the model's prediction degrades. 
While removing important features may affect the prediction of the model, so too can the artifacts introcuded by the removal or perturbation prodecure. 

As a physician we would not want to use any of these methods, either because they lack fidelity or are too slow in point-of-care settings.

## Is There Another Way? 

Recently, *Amortized Explanation Methods (AEM)*  have been introduced to reduce the cost of providing model-agnostic explanations by learning a single global selector model that efficiently identifies the subset of locally important features in an instance of data with a single forward pass. 
AEMs learn the global selector model by optimizing the fidelity of the explainations.

Let's look at the following illustration, which examplifies an AEM:

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/aem.png' | relative_url }}" alt="" title="example AEM"/>
    </div>
</div>
<div class="caption">
</div>

Here the selector model ($q_{\text{sel}}$) plays a game where tries to select features which allow the predictor model ($q_{\text{pred}}$) to predict the target. 
This game is captured by maximizing the following objective:

$$
\mathbb{E}_{x, y \sim F}\mathbb{E}_{s \sim q_{\text{sel}; \beta}(s \mid x ; \beta)}\left[\log q_{\text{pred}}(y \mid m(x, s) ; \theta) - \lambda R(s) \right].
$$

Here selector model ($q_{\text{sel}}$) is optimized to produce selections $s$ that maximize the likehood of the masked data $\log q_{\text{pred}}(y \mid m(x, s) ; \theta)$. 
Then in order to ensure that the explaination is simple, as smaller features selections, the objective pays a penality for selecting each feature expressed as $\lambda R(s)$.


You might be thinking...

## Sounds Great! What's the Catch?

Well, first we have to choose the predictor model.
* Existing Prediction Model: <span style="color:red"> May not work well with the artificats introduced by the masking process (i.e. occulsion to 0.) text</span>
* Train a New Model: <span style="color:red"> Need to be careful.</span>

A few popular *joint amortized explanation methods (JAMs)* such as L2X and INVASE train a new predictor model by learning it jointly with selector model. 
Now the selector and predictor model are playing the game together. 
The selector model tries to select features and the preditor tries to use the masked feature selections to predict the target.

Let's take a look at how this can go horribly wrong:

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/jam_encoding.png' | relative_url }}" alt="" title="JAM encoding"/>
    </div>
</div>
<div class="caption">
</div>

In the above example, we see that JAMs can learn to encode predictions. Here the selector model can select a pixel on the left to indicate dog, and select a pixel on the right to indicate cat. Because, the predictor is trained jointly it can learn these encodings. 
Now, remember that the objective penalizes us for each pixel/feature selection. This encoding solution allows for accurate predictions with just a single pixel selection, helping maximize the objective.

As physicians, being presented with such nonsense will cause us to loose trust in the model. 
We need a way to validate the fidelity of the explainations. 

## Can We Evaluate the Explainations? (EVAL-X)

Well, first we have to choose an evaluator model with which to evaluate the subset of important features identified by the interpretability method 
* Existing Prediction Model: <span style="color:red"> May not work well with the artificats introduced by the masking process (i.e. occulsion to 0.). Restated, explaianations, provided as masked inputs, come from a different distribution than that on which the original model is trained. text</span> 
* Train a New Model: <span style="color:red"> Need to be careful.</span>

Are you getting de-ja-vu?

Most popularly, RemOve And Retrain (ROAR) was introduced to evaluate selections. 
ROAR retrains a model to make predictions from the explainations, provided as masked inputs. 
However, JAMs can encode the prediction directly in th explaination. 
ROAR would simply train a model to learn these encodings, incorrectly validating the explainations. 
Are you getting de-ja-vu, again? 

Instead, we recently introduced EVAL-X. 
Lets look at how EVAL-X works.
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/evalx.png' | relative_url }}" alt="" title="EVAL-X"/>
    </div>
</div>
<div class="caption">
</div>
EVAL-X works by training a new evaluation model to approximate the true probabilty of the target given any subset of features in the input. 
EVAL-X adopts a simple training procedure to learn this model by randomly selecting features during training. 
This procedure exposes the model to the same masking artifacts it will encounter during test time and ensures that the model cannot learn encodings.

## REAL-X, Let us Explain! 

Given that EVAL-X is robust to encodings and out-of-distribution artifacts, you might be wondering... is there a way use this approach to create a new AEM? 
Accordingly, we recently introduced REAL-X a novel AEM!
Lets look at how REAL-X works. (more de-ja-vu)
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/realx.png' | relative_url }}" alt="" title="REAL-X"/>
    </div>
</div>
<div class="caption">
</div>
REAL-X works by first training a new predictor model to approximate the true probabilty of the target given any subset of features in the input using the same proceduer as EVAL-X. 
REAL-X then trains to the selector model to select minimal feature subsets that maximize an aproximation of the true likelihood of the target given any subset of features.
This prevents that the selector model from learning encodings.

REAL-X accomplishes the following:
1. Provides *fast* explainations with a single forward pass
2. Maximizes the explaination *fidelity*
3. Provides *simple* explainations by selecting the minimal set of important features. 

## Do REAL-X and EVAL-X Really Work?

Before we can even think about using REAL-X and EVAL-X in the clinic we need to test the following claims:
1. REAL-X provides fast explainations **without encoding**
2. EVAL-X can **detect encoding** issues

To do so, lets see how REAL-X stacks up againsts other JAMs and see if EVAL-X can detect encodings. Well take a look at:
- L2X (Learning to Explain)
- INVASE 
- BASE-X (Copies REAL-X score function gradient estimation technique REBAR, but is a JAM that learns the selector and predictor models jointly)
- FULL = A model trained on the full feature set.

To make the comparision concrete, our goal is to provide *simple* explainations by selecting as few features as possible while retaining our ability to to predict. 
Restated... 
- We'll tune each method to select the fewest number of features while ensuring that the accuracy (ACC) is within 5% of the original model.

### Evaluation: 

Each AEM we consider first makes selections, then uses those selections to predict the target using is predictor model.
The predictive performance of the AEM is supposed to provide us with a metric of how good the explainations are. 
We'll consider the following metrics: area under the reciever operator curve (**AUROC**) and accuracy (**ACC**).

We'll also look at the predictions that EVAL-X produces given each method's explainations.
Lets denote these metrics with a prefix "e": **eAUROC** and **eACC**.

> If the AEM is encoding, then we would expect *high* AUROC/ACC and *low* eAUROC/eACC.

Now lets see how well each method is able to explain Chest X-Rays.

### Cardiomegally Classification from Chest X-Rays

Cardiomegaly is characterized by an enlarged heart, and can be diagnosed by measuring the maximal horizontal diameter of the heart relative to that of the chest cavity and assessing the contour of the heart. 
Given this, we expect to see selections that establish the margins of the heart and chest cavity.

We used the **The NIH ChestX-ray8 Dataset**
- Subset of 5, 600 X-rays = 2,776 Cardiomegaly and 2,824 Normal
- 5,000: 300: 300 Train, Val, Test Split
- UNet Selector and DenseNet121 Predictor 
- Super-pixel selections
- Training: 50 epochs using a learning rate of .0001
- Tunned the hyperparameter controlling the number of features selected for each method

Let's take a look at some randomly selected explainations from each method. 

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/cxr_1.png' | relative_url }}" alt="" title="CXR_1"/>
    </div>
</div>
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/cxr_2.png' | relative_url }}" alt="" title="CXR_2"/>
    </div>
</div>

<div class="caption">
    The important regions identified by each AEM are over-layed upon each Chest X-Ray in red.
</div>

Based on these samples, L2X, INVASE, and BASE-X are making some pretty strange selections, which don't seem to establish the margins of the heart, the margins of the chest wall, nor the contour of the heart. REAL-X on the other hand seems to be in line with our intuition of what should be important. 

Now, lets see how they perform as mesured directly by the AEM and by EVAL-X:
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/cxr_results.png' | relative_url }}" alt="" title="CXR results"/>
    </div>
</div>
<div class="caption">
</div>
Its clear to see that all the AEMs claim that their explainations are highly predictive based on ACC and AUROC. 
However, EVAL-X is able to reveal that L2X, INVASE, and BASE-X are all encoding the predictions in their explainations, achieving eACC ~50%.
Meanwhile, the slections made by REAL-X remain fairly predictive when evaluated by EVAL-X.

Finally, we had to see what two expert radiologists thought of the explainations generated by each method.
> We randomly selected 50 Chest X-rays from the test set and displayed the selections made by each method for each X-ray in a random order. The radiologists ranked the four options provided.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/cxr_doctor.png' | relative_url }}" alt="" title="CXR doctor"/>
    </div>
</div>
<div class="caption">
    Average rankings by expert radiologists.
</div>

Looks like REAL-X is the #1 physcian recomended explaination method!

## How to Implement REAL-X Yourself?

[Get our code off Github](https://github.com/rajesh-lab/realx)

Explaining with REAL-X invloves three steps: 

1. Initialize the selector model and predictor model. (Any model archetecture can be specified)
2. Choose the REAL-X hyperparameter (lambda) and any other training hyperparameters (i.e. batch_size)
3. Train the predictor and the selector model.

Once REAL-X can been trained, its selector model can be used directly to generate explainations. REAL-X explainations can also be validated with EVAL-X (built-in). 

Please, check out our [example](https://github.com/rajesh-lab/realx/blob/main/example.ipynb) to see how we apply REAL-X to explain MNIST classifications.

### Training REAL-X

This implementation of REAL-X is designed to work with the Keras API.


<d-code block language="python">
    # initialize REALX w/ the selector model, predictor_model, and REAL-X hyperparameter (lambda)
    realx = REALX(selector_model, predictor_model, lambda)
</d-code>

 
<d-code block language="python">
 # train the predictor and selector model
realx.predictor.compile(loss=...,
                        optimizer=...,
                        metrics=...)
realx.predictor.fit(x_train,
                    y_train,
                    epochs=...,
                    batch_size=...)
realx.build_selector()
realx.selector.compile(loss=None,
                       optimizer=...,
                       metrics=...)
realx.selector.fit(x_train,
                    y_train,
                    epochs=...,
                    batch_size=...)
</d-code>

### Generating Explainations with REAL-X
<d-code block language="python">
 # generate explainations
 explainations = realx.select(x_test, batch_size, discrete=True)
</d-code>
 
### Evaluating Explainations with EVAL-X
<d-code block language="python">
 # evaluate explainations with EVAL-X
realx.evalx.compile(loss=...,
                        optimizer=...,
                        metrics=...)
realx.evalx.fit(x_train,
                    y_train,
                    epochs=...,
                    batch_size=...)
y_eval = realx.evaluate(x_test, batch_size)

eAUROC = roc_auc_score(y_test, y_eval, 'micro')
eACC = accuracy_score(y_test.argmax(1), y_eval.argmax(1))
</d-code>