---
layout: distill
title: Learning Accurate ML Explanations with Real-X and Eval-X
description: How do we efficiently generate ML explanations we can trust?
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

bibliography: realx.bib
---


Machine learning models can capture complex relationships in the data that we as humans cannot. 
Perhaps there is a way to explain these models back to humans in order to help humans make high stakes decisions or learn something new. 

When I was working on estimating ejection fraction (a measure of how well the heart pumps blood) using patient electrocardiograms (ECGs), we wanted to understand how our machine learning model was able to estimate ejection fraction. 
I was told that physician’s wouldn’t be able to make such an estimation from ECGs alone, and I thought that if we could explain which parts of the ECG were influencing the model’s decision, then together we could uncover something new. 
However, when attempting to explain my model/data, I ran into a few problems, which I have attempted to capture in the figure below.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/realx_blog.png' | relative_url }}" alt="" title="Why Interpret"/>
    </div>
</div>
<div class="caption">
Understanding ML models can expand clinical knowledge. If we have a model that estimates heart functioning from a patient's ECG, understanding how/why can help expand our understanding of electrophysiology.
</div>

Firstly, ECGs are highly variable and complex, so we needed unique, personalized explanations for each ECG, which was not a quick process.
However, my biggest issue was that I was in unknown territory and couldn't be sure if any of the explanations generated were reasonable or could be trusted.
Trusting the model was important, but trusting the explanation was even more important. 

As pictured above, explanations identify which regions of the ECG are "important" using some mathematical operation.
However, I wasn't sure if these mathematical operations could be translated into something clinically relevant. 
For example, are large gradients for certain regions of the ECG clinically significant?
Further, does the data support, with high fidelity, the conclusion that the important regions identified by the explanation method drive the estimation of ejection fraction?


### Why Do We Need Interpretability in Machine Learning?

Data is complicated. 
Often it is difficult to make sense of complex patterns in the data, because the process by which it is generated remains unclear. 
Instead, practitioners have turned to machine learning to help model these mechanisms. 
Unfortunately, ML models are often too complex to understand. 

In medicine, we are beginning to see ML models perform tasks that physicians cannot. 
For example, as discussed, estimating estimating ejection fraction using ECGs<d-cite key="Attia2019a"></d-cite> or predicting 60 day mortality<d-cite key="major2020estimating"></d-cite>. 
Understanding how ML models can generate their predictions can help expand clinical knowledge. 

It's natural to then ask...

### What is an Interpretation/Explanation?

At a high level, explanations aim to provide users with an idea of what information is important for generating the model's prediction or the target of interest.

One of the most active area of research in ML interpretability looks to provide users with explanations for a single prediction/target, by scoring the influence of each feature in the input.
However, it is important to consider what these scores mean when translating them to an explanation.

Each interpretability method defines some mathematical operation, which provides their definition of interpretability. 
For example, take gradient-based explanations.
Many of these, estimate the gradient of the prediction/target with respect to the input.
Roughly, the gradient estimates how that model's output may change under very small perturbations to the input. 
While, this may be useful, it is important to note that this is different from the identification of which features are the most important for generating the prediction/target.
Recent work has shown this empirically<d-cite key="NIPS2019_9167"></d-cite>.

<!-- 
A physician may want to understand which genes are involved in diabetes. However, often physicians want more personalized information about their patient. Instead, it would be more useful to understand which of their patient's genes are contributing to their diabetes so they can more effectively develop treatment plans. 

Accordingly, most interpretability methods focus on providing local explanations. Local or instance-wise explanations provide reasons for why a specific decision was made. This is often accomplished by providing the feature importances related to that decision. -->

Now let's answer...

### What Do We Want in an Explanation?

Physicians, for example, need to be able to make quick, accurate decisions in order to treat patient's effectively. Based on our experience working with ECGs, we complied the follow list of wants, alluded to in the figure above:

1. Instancewise (aka Personalized)
- We want instancewise explanations for multiple reasons. Firstly, our data may be translationally invariant. For example, we may be interested in identifying cancer from histology images. Here, our explanation should identify cancerous cells regardless of their location within the image. Secondly, instancewise explanations allow for a more granular understanding of the data. In medicine, this may help provide personalized information.  
1. Accurate/ High Fidelity
- We want explanations that can reliably provide which features are the most important for generating the model prediction/target. 
3. Simple
- We want explanations that are simple enough to be interpreted by a human, both in terms of their presentation to a human and the mathematical concept conveyed by the explanation. 
4. Fast
- Providing a granular/personalized understanding of the data can be computationally challenging, however it allows for a more dynamic insights to be gained. Therefore, we want to be able to provide explanations that scale to large datasets and can be accessed in real-time. 

Now that we know we want...

### Can We Have What We Want?

Let's consider existing interpretability methods and break them into three groups: 

|   | Gradient Based Methods    | Locally Linear Methods    | Perturbation Methods  |
|:-:    |:-:    |:-:    |:-:    |
| What?     | Measure the gradient of output   with respect to the input features?    | Uses a linear function of  simplified variables to explain  the prediction of a single input  | Perturb the inputs and observe the effect on the target    |
| Examples  | gradCAM<d-cite key="Selvaraju_2019"></d-cite>, DeepLift<d-cite key="Shrikumar2017"></d-cite>   | LIME<d-cite key="Ribeiro2016"></d-cite>, SHAP<d-cite key="Lundberg2017"></d-cite>    | Occlusion<d-cite key="Zeiler2014"></d-cite>     |
| Why (not)?    | Explanations don't optimize for accuracy/fidelity. Recent work shows estimates of feature importance often do not identify features that help predict the target<d-cite key="NIPS2019_9167, adebayo2018sanity"></d-cite>.     | These methods are slow, requiring numerous perturbations of the input and/or training a new model per explanation. These perturbations may evaluate models where they are not grounded by data.    | These methods are slow, requiring numerous perturbations to generate a single explanation. These perturbations may evaluate models where they are not grounded by data.      |

Of note, both locally linear and perturbation-based methods rely on removing or perturbing features in order to characterize how/if the model's prediction degrades. 
While removing important features may affect the prediction of the model, so too can the artifacts introduced by the removal or perturbation procedure. 

While we can think of reasons to use each of these methods, none of them seem to satisfy our list of wants, either because they lack fidelity to the data or are too slow to scale to large datasets.

### Is There Another Way?

Recently, Amortized Explanation Methods (AEM)<d-cite key="Dabkowski2017, Chen2018, yoon2018invase, schwab2019cxplain"></d-cite> have been introduced to reduce the cost of providing model-agnostic explanations by learning a single global selector model that efficiently identifies the subset of important features in an instance of data with a single forward pass. 
Amortized explanation methods learn the global selector model by optimizing the fidelity of the explanations to the data.

Let's look at the following illustration, which exemplifies an amortized  explanation method :

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/aem.png' | relative_url }}" alt="" title="example AEM"/>
    </div>
</div>
<div class="caption">
</div>

Here the selector model ($q_{\text{sel}}$) plays a simple game where it tries to select features which allow the predictor model ($q_{\text{pred}}$) to predict the target.
This game aims to maximize the fidelity of the explanations directly. 
This game is captured by maximizing the follow amortized explanation method objective:

$$
\mathcal{L}_{AEM} = \mathbb{E}_{x, y \sim F}\mathbb{E}_{s \sim q_{\text{sel}; \beta}(s \mid x ; \beta)}\left[\log q_{\text{pred}}(y \mid m(x, s) ; \theta) - \lambda R(s) \right].
$$

Here selector model ($q_{\text{sel}}$) is optimized to produce selections $s$ that maximize the likelihood of the masked data $\log q_{\text{pred}}(y \mid m(x, s) ; \theta)$. 
Then in order to ensure that the explanation is simple (presents a small set of important features) the objective pays a penalty for selecting each feature, expressed as $\lambda R(s)$.

You might be thinking...

### What's the Catch?

Well, first we have to choose the predictor model.
 We can either use an existing prediction model, which <span style="color:red"> may not work well with the artifacts introduced by the masking process (i.e. occlusion to 0.) text</span>, or train a new model, which <span style="color:red"> requires care.</span>

A few popular joint amortized explanation methods (JAMs) such as L2X<d-cite key="Chen2018"></d-cite> and INVASE<d-cite key="yoon2018invase"></d-cite> train a new predictor model by learning it jointly with selector model. 
Now the selector and predictor model are playing the game together. 
The selector model tries to select features and the predictor tries to use the masked feature selections to predict the target.

Let's take a look at how this can go wrong:

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/jam_encoding.png' | relative_url }}" alt="" title="JAM encoding"/>
    </div>
</div>
<div class="caption">
</div>

In the above example, we see that these joint amortized explanation methods can learn to encode predictions. 
Here the selector model can select a pixel on the left to indicate dog and select a pixel on the right to indicate cat. 
Because the predictor is trained jointly, it can learn these encodings. 
Now, remember that the objective penalizes us for each pixel/feature selection. 
This encoding solution allows for accurate predictions with just a single pixel selection, helping to maximize the amortized objective.

Presented strange explanations like this in clinical settings can lead physicians to quickly loose trust. 
We need a way to validate the fidelity of the explanations. 

### Can We Evaluate the Explanations? (Eval-X)

Well, first we have to choose an evaluator model with which to evaluate the subset of important features identified by the interpretability method.
We can either use an existing prediction model, which <span style="color:red"> may not work well with the artifacts introduced by the masking process (i.e. occlusion to 0.), </span> or train a new model, which <span style="color:red"> requires care.</span>

Are you getting de-ja-vu?

Popularly, RemOve And Retrain (ROAR)<d-cite key="NIPS2019_9167"></d-cite> was introduced to evaluate selections. 
ROAR retrains a model to make predictions from the explanations, provided as masked inputs. 
However, the joint training procedure can encode the prediction directly in the explanation. 
ROAR would simply train a model to learn these encodings, incorrectly validating the explanations. 
Are you getting de-ja-vu, again? 

Instead, we recently introduced Eval-X. 
Lets look at how Eval-X works.
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/evalx.png' | relative_url }}" alt="" title="Eval-X"/>
    </div>
</div>
<div class="caption">
</div>
Eval-X works by training a new evaluation model to approximate the true probability of the target given any subset of features in the input. 
Eval-X adopts a simple training procedure to learn this model by randomly selecting features during training. 
This procedure exposes the model to the same masking artifacts it will encounter during test time and ensures that the model cannot learn encodings.

### Real-X, Let us Explain! 

Given that Eval-X is robust to encodings and out-of-distribution artifacts, you might be wondering... is there a way use this approach to create a new amortized explanation method? 
Accordingly, we recently introduced Real-X, a novel amortized explanation method!
Lets look at how Real-X works. (more de-ja-vu)
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/realx.png' | relative_url }}" alt="" title="Real-X"/>
    </div>
</div>
<div class="caption">
</div>
Real-X works by first training a new predictor model to approximate the true probability of the target given any subset of features in the input using the same procedure as Eval-X. 
Real-X then trains a selector model to select minimal feature subsets that maximizes the likelihood of the target, as measured by the Eval-X style predictor model.
This prevents the selector model from learning encodings.

Real-X accomplishes the following:
1. Provides fast explanations with a single forward pass
2. Maximizes explanation fidelity/accuracy
3. Provides simple explanations by selecting the minimal set of important features. 

### Do Real-X and Eval-X Really Work?

Before we can even think about using Real-X and Eval-X in the clinic we need to test the following claims:
1. Real-X provides fast explanations without encoding
2. Eval-X can detect encoding issues

To do so, lets see how Real-X stacks up against other amortized explanation methods and whether or not Eval-X can detect encodings. 
Well take a look at:
- L2X (Learning to Explain)<d-cite key="Chen2018"></d-cite> 
- INVASE<d-cite key="yoon2018invase"></d-cite>
- BASE-X (Copies Real-X score function gradient estimation technique REBAR, but is a an amortized explanation method that learns the selector and predictor models jointly)
- FULL = A model trained on the full feature set.

To make the comparison concrete, our goal is to provide simple explanations by selecting as few features as possible while retaining our ability to predict. 

<!-- Fundamentally, theres a trade-off between how simple the explanation in term of the number of features selected and how much these features help with the prediction problem.
This is a choice the practitioner has to make.
- We chose to tune each method to select the fewest number of features while ensuring that the accuracy (ACC) is within 5% of the original model. -->

#### Evaluation: 

Each amortized  explanation method  we consider first makes selections, then uses those selections to predict the target using is predictor model.
The predictive performance of the amortized explanation method is supposed to provide us with a metric of how good the explanations are. 
We'll consider the following metrics: area under the receiver operator curve (AUROC) and accuracy (ACC).

We'll also look at the predictions that Eval-X produces given each method's explanations.
Let's denote these metrics with a prefix "e": eAUROC and eACC.

> If the amortized  explanation method  is encoding, then we would expect high AUROC/ACC and low eAUROC/eACC.

Now lets see how well each method is able to explain Chest X-Rays.

#### Cardiomegaly Classification from Chest X-Rays

Cardiomegaly is characterized by an enlarged heart and can be diagnosed by measuring the maximal horizontal diameter of the heart relative to that of the chest cavity and assessing the contour of the heart. 
Given this, we expect to see selections that establish the margins of the heart and chest cavity.

We used the The NIH ChestX-ray8 Dataset<d-cite key="Wang_2017"></d-cite>

<!-- - Subset of 5, 600 X-rays = 2,776 Cardiomegaly and 2,824 Normal
- 5,000: 300: 300 Train, Val, Test Split
- UNet Selector and DenseNet121 Predictor 
- Super-pixel selections
- Training: 50 epochs using a learning rate of .0001
- Tunned the hyperparameter controlling the number of features selected for each method -->

Let's take a look at some randomly selected explanations from each method. 

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/cxr_2.png' | relative_url }}" alt="" title="CXR_2"/>
    </div>
</div>
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/cxr_1.png' | relative_url }}" alt="" title="CXR_1"/>
    </div>
</div>

<div class="caption">
    The important regions identified by each explanation method  are over-layed upon each Chest X-Ray in red.
</div>

An initial review of these samples suggests that L2X, INVASE, and BASE-X may be making some selections that don't appear to establish the margins of the heart, the margins of the chest wall, nor the contour of the heart. 
Real-X on the other hand appears to be in line with our intuition of what should be important.
However, we can't be sure without additional evaluation. 

Now, lets take a look at the in-built and EVAL-X evaluation metrics:
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/cxr_results.png' | relative_url }}" alt="" title="CXR results"/>
    </div>
</div>
<div class="caption">
</div>
All the explanation methods provide explanations that are highly predictive when assess directly by the method. 
However, Eval-X is able to reveal that L2X, INVASE, and BASE-X are all encoding the predictions in their explanations, achieving eACC ~50%.
Meanwhile, the sections made by Real-X remain fairly predictive when evaluated by Eval-X.

Finally, let’s look at what two expert radiologists thought of the explanations generated by each method.
> We randomly selected 50 Chest X-rays from the test set and displayed the selections made by each method for each X-ray in a random order. The radiologists ranked the four options provided.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/cxr_doctor.png' | relative_url }}" alt="" title="CXR doctor"/>
    </div>
</div>
<div class="caption">
    Average rankings by expert radiologists.
</div>

From this, we see that the physicians tended to choose the selections generated by Real-X.


### How to Implement Real-X?

[Get our code off Github](https://github.com/rajesh-lab/realx)

Explaining with Real-X involves three steps: 

1. Initialize the selector model and predictor model. (Any model architecture can be specified)
2. Choose the Real-X hyperparameter (lambda) and any other training hyperparameters (i.e. batch_size)
3. Train the predictor and the selector model.

Once Real-X can been trained, its selector model can be used directly to generate explanations. Real-X explanations can also be validated with Eval-X (built-in). 

Please, check out our [example](https://github.com/rajesh-lab/realx/blob/main/example.ipynb) to see how we apply Real-X to explain MNIST classifications.

<!-- #### Training Real-X

This implementation of Real-X is designed to work with the Keras API.


<d-code block language="python">
    # initialize REALX w/ the selector model, predictor_model, 
    # and Real-X hyperparameter (lambda)
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

#### Generating Explanations with Real-X
<d-code block language="python">
 # generate explanations
 explainations = realx.select(x_test, batch_size, discrete=True)
</d-code>
 
### Evaluating Explanations with Eval-X
<d-code block language="python">
 # evaluate explanations with Eval-X
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


<script type="text/bibliography">
  @article{gregor2015draw,
    title={DRAW: A recurrent neural network for image generation},
    author={Gregor, Karol and Danihelka, Ivo and Graves, Alex and Rezende, Danilo Jimenez and Wierstra, Daan},
    journal={arXivreprint arXiv:1502.04623},
    year={2015},
    url={https://arxiv.org/pdf/1502.04623.pdf},
  }
</script> -->