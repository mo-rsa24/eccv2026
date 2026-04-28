# Slide Title

How The MDS Trajectory Plot Is Actually Constructed

## Slide Subtitle

From prompts and shared noise to a 2D `sklearn` projection

## On-Slide Text

- Step 1: define the conditions from the same concept pair
  - `A`
  - `B`
  - `A∧B` (monolithic prompt)
  - `PoE` (logical composition)
- Step 2: encode each prompt with the model text encoder(s)
  - SD1.x: tokenizer + text encoder
  - SD3.5: tokenizer/text encoder stacks used by the script
- Step 3: start all conditions from the same initial latent noise `x_T`
- Step 4: run denoising and record the latent at every timestep
- Step 5: flatten each recorded latent trajectory into vectors and stack all conditions together
- Step 6: compute a pairwise Euclidean distance matrix over all trajectory points
- Step 7: run `sklearn.manifold.MDS` on that precomputed distance matrix
- Step 8: plot the resulting 2D trajectories; decode final latents with the VAE only for the image strip

## Suggested Slide Layout

- Use a vertical pipeline or left-to-right flow diagram.
- Put the 8 steps in the main body.
- Put the equations at the bottom.
- Add one small note box: `VAE is for decoding endpoint images, not for building the MDS coordinates`.

## Figures To Include

- Optional reuse:
  - `paper/neurips/Comparing Semantic and Logical Composition Using Latent Diffusion Models/figures/trajectory_2x2_sdipc.png`

## Figure Usage Notes

- If you reuse the trajectory figure, use it as a small reference thumbnail only.
- The main visual should be the pipeline diagram, not the figure itself.
- Add a callout:
  - `same x_T for all conditions`
  - `2D coordinates come from sklearn MDS on latent trajectories`

## Mathematical Notation / Equations

Conditions:

$$
\mathcal{C} = \{A,\; B,\; A \wedge B,\; \mathrm{PoE}\}
$$

Shared initial latent:

$$
x_T \sim \mathcal{N}(0, I)
$$

with the same sampled $x_T$ reused across all conditions.

Recorded trajectories:

$$
\mathcal{Z}^{(c)} = \{ z_t^{(c)} \}_{t=0}^{T}, \qquad c \in \mathcal{C}
$$

Flatten each latent state:

$$
f_t^{(c)} = \mathrm{vec}\!\left(z_t^{(c)}\right) \in \mathbb{R}^{D}
$$

Build one stacked matrix across all conditions and all timesteps:

$$
F = \begin{bmatrix}
f_0^{(A)} \\
\vdots \\
f_T^{(A)} \\
f_0^{(B)} \\
\vdots \\
f_T^{(\mathrm{PoE})}
\end{bmatrix}
$$

Compute pairwise Euclidean distances:

$$
D_{ij} = \left\|F_i - F_j\right\|_2
$$

Then project with `sklearn.manifold.MDS`:

$$
Y = \mathrm{MDS}(D), \qquad Y \in \mathbb{R}^{N \times 2}
$$

Finally, split $Y$ back into one 2D trajectory per condition.

VAE decoding happens only for endpoint rendering:

$$
\hat{x}^{(c)} = \mathrm{VAE.decode}\!\left(z_T^{(c)}\right)
$$

## Speaker Notes

- This is the actual pipeline in the repo.
- The relevant implementation points are:
  - shared-noise latent generation via `get_latents(...)`
  - prompt encoding via the text encoder stack inside `trajectory_dynamics_experiment.py`
  - trajectory tracking with `LatentTrajectoryCollector`
  - flattening and stacking in `project_trajectories(...)`
  - pairwise distances via `sklearn.metrics.pairwise_distances`
  - 2D projection via `sklearn.manifold.MDS`
- Important clarification:
  - the MDS coordinates are built from latent tensors during denoising
  - the VAE is not used to compute the MDS plot
  - the VAE is only used afterward to decode final latent endpoints for visual inspection

## Repo Anchors

- `scripts/trajectory_dynamics_experiment.py`
- `notebooks/dynamics.py`
- `notebooks/utils.py`
