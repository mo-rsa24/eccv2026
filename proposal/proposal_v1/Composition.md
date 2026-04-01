Paper: Compositional Generative Modeling A Single Model is Not All You Need
Section 2.3
Authors (cite) argue that we should instead construct large generative systems by composing smaller generative models together.

Provide an argument for why the following statement is helpful to the task of leveraging TB and silicosis data to generate OOD data to improve downstream tasks.
Also mention how extensible it would be, in general, to other medical imaging domains and co-morbid pathologies: 
"we should instead construct complex
generative models compositionally from simpler models."


Each constituent model captures the probability distribu-
tion of a subset of variables of the distribution of interest,
which are combined to model the more complex full distribu-
tion. Individual distributions are therefore much simpler and
computationally modeled with both fewer parameters and
learnable from less data. Furthermore, the combined model
can generalize to unseen portions of the data distribution as
long as each constituent dimension is locally in distribution


There is a general method to composition

There are three fundamental compositional operators: Product(AND), Mixture(OR), Negation(Not)

1. Product(AND)
Multiply probabilities from multiple models:

Each model states a constraint (“what must be true”). The product keeps only outcomes satisfying all constraints.

Mention here that Liu et all 2022 characterize a single generative model where they factor the generation as a product of distribution. 
Show their P(x |T ) prop p(x|t1) p(x|t2) etc
mention that it is more data efficient because we only need to see full distribution of images given single setnences. Additionally 
allows them to generalize to unseen regions of p(x|T) such as unseen combinations.
Likewise, in the experiments of Skreta, they use an SDXL which is a single generative model and factor generation as product of distribution as well.
However, we want to be able to have two models

2. Mixture (OR)


3. Negation (NOT)


Various ways in which you can apply these operators:

Generative Modeling with Learned
Compositional Structure

  A limitation of compositional generative modeling dis-
  is that it requires a priori knowledge about the independence structure of the distribution we
  wish to model

  unsupervised latent composition: discover factors and you compose them
  The underlying compositional components in generative
  modeling can in many cases be directly inferred and dis-
  covered in an unsupervised manner from data, representing
  compositional structure such as objects and relations. Such
  discovered components can then be similarly recombined
  to form new distributions – for instance, objects compo-
  nents discovered by one generative model on one dataset
  can be combined with components discovered by a separate
  generative model on another dataset to form hybrid scenes
  with objects in both datasets. We illustrate the efficacy of
  such discovered compositional structure across domains in
  images and trajectory dynamics

  Factorization / Structured Composition


Factorized 

Research Method

The basic method is that we have a target distribution pSTB(x) which we want to factorize Silicosis & TB.
Mention explicitly with citations that compositional generative modeling, superdiff included, does not necessitate shared latent representations.
In fact, we are trying to prove and demonstrate that semantic compositional requires latent alignment whereas theoretically there is no need for it.

Factors ps(x) and pTB(x).


We argue that Silico-TB should be modeled as explicit / a-priori compositional structure, not as learned compositional structure.
This is because:

We already know the meaningful factors (pathologies, anatomy).

We do not want the model to discover arbitrary latent factors (orientation, marker artifacts, texture statistics) because these are not the medically relevant axes of variation.

Incorrect latent discovery could entangle or miss pathologies, leading to compositional errors.

Thus silico-TB falls under the “Factorization / Structured Composition” category, not “Learned Compositional Structure.” based on this sentence 
"If we know that a distribution exhibits an independence structure… we can substantially reduce the data requirements by learning factors and composing them." from the paper


Why not learned compositional structure?

Section 4 of the paper describes unsupervised discovery:
- objects
- relational forces
-classes,

or other latent components.

But these discovered components are not guaranteed to align with clinically meaningful factors.

They often correspond to:
- pose,
- lighting,
- noise/artifacts,
- view angle,
- dominant texture directions,
- machine-specific variation,
- irrelevant confounders.

The paper itself warns about this:

“Current work on discovering compositional structure assumes data is naturally factorized… real data often exhibits spurious correlations… causing algorithms to fail to discover the correct structure.” (Section 6, limitations).

This is exactly what you want to avoid in a medical setting.



Figure 3 In Compositional Generative MOdeling: A Single Model Is Not All you Need
Make our case around this "Generalizing Outside Training Data. Given a narrow
slice of training data, we can learn generative models that gen-
eralize outside the data through composition. We learn separate
generative models to model each axis of the data – the composition
of models can then cover the entire data space."


Also
Figure 4. Distribution Composition – When modeling simple
product (top) or mixture (bottom) compositions, learning two com-
positional models on the factors is more data efficient than learning
a single monolithic model on the product distribution. The mono-
lithic model is trained on twice as much data as individual factors

Because We dont know yet which one Silico-Tb will fall under


Experiments:
Test composition and mixture. Basically recreate figure 4. 

Limitations to be aware of:
"Similarly, two
learned score functions from diffusion models are not di-
rectly composable as they do not correspond to the noisy
gradient of the product distribution"

While it is often difficult to combine generative models,
representing the probability density explicitly enables us
to combine models by manipulating the density. One such
approach is to represent probability density as an Energy-
Based Model,