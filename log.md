## 2026-04-27

### What I did
- Read the DeepDeWedge paper
- Made some sketches in 2D and 3D of my intuition about the missing wedge in the fourier space.
- Found a nice simulation online: Fourier Slice theorem illustration
  https://www.youtube.com/watch?v=4aDoKzqfCaM
- setup the conda environment on the jupyterhub (euler cluster)
- follow step0 to get my first tomograms. Noticed the blurriness in the XZ and YZ slices. 
- setup the github repo and created a clear directory structure

### What I tried / what failed
- First misunderstood the axis convention in the Fourier - tomography / imaging setup (still unsure, GPT and Claude supported my last notation of $k_x, k_y, k_z$)
- problems with using the euler cluster through the terminal and sending jobs

### Technical details & thoughts
- I clarified that Fourier space has the same dimensionality as the real-space object, but it does not have the same geometric shape. A 3D cuboid object does not become a 3D cuboid in Fourier space; instead, its Fourier transform describes spatial frequency content.
- I clarified how a projection is calculated. For example, if $f(x,y)$ is a 2D density function, then a projection along the $y$-direction is obtained by integrating over $y$ for each $x$:
  $$
  p(x) = \int_{-\infty}^{+\infty} f(x,y)\,dy.
  $$
  The 1D Fourier transform of this projection corresponds to a line through the 2D Fourier transform of the object. In 3D, the analogous statement is that the 2D Fourier transform of a projection corresponds to a plane through the 3D Fourier transform.
- In cryo-ET, the starting object is a 3D volume $f(x,y,z)$. A projection at tilt angle $\theta$ gives a 2D image $p_\theta(x,y)$. Its 2D Fourier transform $\hat p_\theta(k_x,k_y)$ corresponds, by the Fourier Slice Theorem, to a central 2D slice through the 3D Fourier transform $\hat f(k_x,k_y,k_z)$.
- Because the tilt range is limited, e.g. to approximately $\pm 60^\circ$ instead of $\pm 90^\circ$, some Fourier planes are never measured. This leaves wedge-shaped missing regions in 3D Fourier space, which leads to anisotropic resolution and blurring/elongation especially in directions involving $z$.


## 2026-04-28

### What I did
- Setup the Step1 jupyter notebook from the tutorial
- Analyzing "prepare_data" parameters and "shared" parameter to understand what I can adjust and what is given as input
- Plotted 5 sub-tomogram pairs (even frame vs odd frame) side by side
- read fourier.py, understanding of the fft and iftt, dimensions and mask multiplication

### What I tried / what failed
- !conda run -n ddw_env ddw prepare-data → conda: command not found
- !ddw prepare-data → ddw: command not found
- Fixed: Euler's Jupyterhub shell has a minimal PATH, does not include the conda env.
- Had to workaround this by looking for the correct directory via sys.executable. This is in the same dir as the needed ddw. get the directory name and join the ddw path in order to use 'prepare-data'.
- Tried editing kernel.json, failed miserably and lost it, had to register again
- Struggled with the understanding of the mask in missing_wedge.py¨

### Technical details & thoughts
- Apparently my conda environment is setup correctly after checking with AI.
- Understanding the data input, the tilt series t is already split up and filtered back-projection (FBP) is applied. We simply load the two different tomogram data. Dimension analysis and understanding how to get different slices from the v0 and v1 volumes.
- 144 sub-tomograms saved for fitting, 36 for validation (80/20 split)
- after sketching the plane normal, realized k=(kx, ky, kz) cannot be true. Analyzed the dimensions of the tomogram. Found out:
tomo[half[0], :, :]  = X-Y slice  → axis 0 is z
tomo[:, half[1], :]  = X-Z slice  → axis 1 is y
tomo[:, :, half[2]]  = Y-Z slice  → axis 2 is x.
Thus k = (kz, ky, kx)




## 2026-05-04

### What I did
- Read and analyzed the subtomo_dataset.py and masked_loss.py
- Understanding of how the masks work, how the loss calculation uses the different masks to make sure no region without actual information gets used for loss calculation.
- Answering Questions from day three plan.
- Complete sketch of the U-Net architecture, encoder and decoder with skips.
- Reading and understanding the unet.py


### What I tried / what failed
- Tried to setup my jupyterhub server with GPUs, this did not work! Need to ask Yves Acremann on how to get permission for GPU access on ETH Euler.
- Connection problems, recorded in the error directory
- Running the fit-model on GPUs from Euler Cluster, My account does not have access yet to GPUs. That is why everything worked until now but I could not access JupyterHub with GPUs. Currently in contact with IT support of Euler Cluster because of this. Could not launch the training.

### Technical details & thoughts
- Q1: Why are sub-tomograms extracted at √2 × target size?
- A1: Because a random rotated cube needs more space than the original cube, to account for this and not loose information at the boundary depending on the rotation we extract a bigger volume for the initial volume. Than later crop it down to target size.
- Q2: How does random 3D rotation work, and why does it help?
- A2: Both subtomo0 and subtomo1 get randomly rotated (same rotation for both), this helps s. t. our network does not just learn the reconstruction in one direction.
- Q3: What exactly gets fed into the model vs. what is the training target?
- A3: rotated subtomo0 + the artificial wedge (mask) is used as model input. rotated subtomo1 is used as the model target.
- Q4: Why is the loss computed only over the missing wedge region? 
- A4: This is wrong in my opinion, the loss is calculated everywhere except the original missing wedge region. There exist two regions and they are differently weighted. (outside and inside artificial mask)
- Q5: What would happen if you computed loss everywhere?
- A5: We would use regions where we do not have a ground truth as a indicator for our model (loss), makes no sense since this region does not have good or complete information that would be helpful for reconstruction.






## 2026-05-05

### What I did
- Email contact with IT Support from Euler Cluster, could not help -> mail towards ETH D-PHYS and Yves Acremann (ETH Supervisor)
- Read and analyzed fit_model.py, answered the questions and built a deep understanding of how our training works.
- read and analyzed subtomos.py creation of subtomos out of the huge raw tomograph data. (cubic window slicing) answered the questions aswell
- read and analyzed normalization.py how when and why we use normalization. Smart approach behind the normalization.
- Wrote a short note explaining the full data flow. Was crucial to understand how each of the .py files work and play together. Understanding when and where exactly we using what. (Gave me a solid base about the code structure)


### What I tried / what failed
- Still no access on Euler Cluster for accessing GPUs

### Technical details & thoughts






## 2026-05-11

### What I did
- setup the repo on my computer and created the conda environment locally on my computer
- setup my own GPU such that I can use it for training purpose. 
- ran the step 3 with the provided pre-trained checkpoint (fitted_model.ckpt) and it worked (~ 2 hrs runtime)
- One could see the reduced blurring in the comparison of refined tomogram and the original one
- Got ra-cluster access without VPN access, setup my desktop and started with the conda environment again
- Started the refinement process on the ra cluster aswell (step03 tutorial), it worked with GPU access. Setup should now be ready.
- comparison figure for the input vs refined (saved as png, visible in figures directory)
- reading and understanding  refine_tomogram.py

### What I tried / what failed
- tried to get access to the ra-cluster with VPN-access but I could not get VPN access since this is not possible for external users at PSI. 

### Technical details & thoughts
- learned how to use my own GPU for ML (setup and allocation)
- connection to ra-cluster, getting to know linux and linux terminal. Starting the .py files from linux terminal.
- source /scratch/ext-schnei_f/envs/ddw_env/bin/activate (to activate the ddw__env environment on the ra cluster in the morning)



## 2026-05-12

### What I did
- started the training from day 3 on the ra cluster. "launch training: ddw fit-model --config config.yaml"
- currently at epoch 80 from 200, I checked such that the metrics.csv is actually tracking and I later can look at the fitting loss and validation loss decrease
- setup a notebook for the FSC function and ran some simple comparison from an even and odd tomogram
- computed and plotted the 3D spectrum of input and refined tomogram, compared the middle slize (2D) and one can see how the missing wedge gets filled up.
- setup jupyter notebook for the mw angle sweep
- ra cluster training is at 75% at the end of the day


### What I tried / what failed
- used the wrong training setup on the ra cluster, when I lost the connection it stopped the training. Had to restart with some tmux command such that it runs independently of my connection.
- running the refinement proccess on my microsoft surface failed miserably, jupyter notebook never finished with the cell, could not check if something is done in the background or if it was just stuck...

### Technical details & thoughts
- Tried using my laptop for the refinement mw angle sweep part, did not work! Need either my GPU from my computer (was not at home today) or the ra-cluster (which was occupied with the training). Will try to run this mw angle sweep tomorrow and see if the outputs are correct.

### MISSING
Afternoon of day 6 and actual running all of the inference steps



## 2026-05-13 (extra day)

### What I did
- Day 4 Training Loss Analysis jupyter notebook where I plotted fit_loss and val_loss. Overall decrease is visible and both converge against the same loss. fit_loss is oscillating more around (more noise)
- val_loss has less values but also less noise
- started the refinement process on the ra cluster. (mw_angle_sweep)
- jupyter notebook for the FSC comparison (ground truth was tomo _all_frames.rec)
- compared and plotted the 5 different FSC for the 5 different mw angles
- Calculated the FSC-0.5 threshold condition


### What I tried / what failed
- not sure if my funcitons are correct, since my outcome seems strange or does not align with the given table setup


### Technical details & thoughts
- training finished after 200 epochs. Around 16 hrs runtime. Everything worked well, log files all here and look complete.
- still took almost 2 hours for 5 refinements (on ra cluster)




## 2026-05-18

### What I did
- got access to /das/work/pgroup
- environment setup there such that my environment or data wont get deleted all the time
- updated the day 6 FSC notebook such that I actually calculate the resolution aswell. Resolution given through the wavelength. Basically as long as some FSC for a frequency is correlated (close to 1), most likely these means we are capturing real structure since the input tomos are independent. So our model actually learned the same structure, if we end up with 0 correlation this means we are looking at noise.



### What I tried / what failed
- lost my progress on the ra-cluster since I was working on my home directory, there the 20GiB limitation forced me to use the /scratch/. After not using the connection for 3 days it got terminated and only part of the environment on the home/ext-schnei_f/ was not deleted.
- tried getting access to /das/work/ directory but I did not have access yet.
- also realized that my FSC comparison of day 6 is wrong. I do not calculated any resolutions and compare them. I did compare the single refined tomograms for a fixed mw to the so called "ground truth" in the plan (all_frames.rec) which actually is not a ground truth, just even + odd and still has noise + missing wedge



### Technical details & thoughts
-setting up correct working space on a cluster is important
-also AI told me not to push any .cpkt .csv .rec and so on, but after the deletion of my ra cluster, I lost all these files because were not pushed to github (AI told me too much storage) 
- Would def. to this different in the future, this was annoying but very helpful. Better to learn / experience this now than in the future.



## 2026-05-19

### What I did
- setup my conda environment on the p23/p23467 group. Also made some directory such that I can use this for the big files I train/refine.
- uploaded the data to the ra-cluster (even/odd/mask/model etc) needed for the refinement.
- copied the refine_tomogram.py and made a new .py file where we do not average over even and odd but rather extract the even and odd tomograms. Which we need for FSC gold standard.
- Tried refinement with the new .py file (currently running)
- switched to batchsize = 1 since I got errors (to much memory needed, more than was free) when starting the refinement with batchsize= 5 or 10.

### What I tried / what failed
- connection problems with ra cluster during the morning
- refinement with batchsize = 5 and batchsize = 10, failed because of not enough free memory
- uploading files through nomachine failed, no reason why, could not use my mouse to access the upload menu

### Technical details & thoughts
- had a lot of technical problems today, mouse wasnt working in nomachine, no solution found currently
- nomachine connection did not work
- /das/home always gets loaded with .cache or .conda directorys which use up all of the free space I have left on my home/ext-schnei_f/, do not know why, the conda env. is setup on das/work/p... so really confusing.


## 2026-05-21 (extra day 2)

### What I did
- tomograms did refine, got even, odd and even + odd for each mw_angle.
- Used my jupyter notebooke for day 6 and got the FSC for the same angle but compared even and odd.
- Saw that with increasing angle the FSC increased. Such that for mw=30° it was the worst. Especially towards the higher frequencies.
- started the model training for 40°, 50° and 60° with batchsize=2, num_epoch=200.
- finished subexp01, the power spectrum analysis and the XZ, XY comparison ratios.


### What I tried / what failed
- Had again problems with using the download from the nomachine, my mouse was not correctly working. Could not click the bottom right menu used for the download.
- training the model with batchsize = 5, always out of memory error, always stopped the training.
- switched to batchsize = 2, where it worked again, also 3 and 4 did not work, same memory error.


### Technical details & thoughts
- Whenever I work on nomachine on my laptop I get problems with inputs from mouse and sometimes keyboard, tried using another mouse (not touchpad) but did not really help. When working from my stationary computer at home most of the things than work, I dont know if its just the microsoft surface that is giving me input problems or what exactly the problem is...



## 2026-05-26 

### What I did
- finished 2D Fourier Slice comparison for 30°, 50° and 70°
- analyzed under- and overfitting. Defined the boundaries to compare the different regions.
- saved the plots for report purpose, wrote explanation in notes.md
- finished training of all 3 different models (40°, 50°, 60°, unfortunately only with batchsize = 2)
- notebook for loss analysis (for training the three different models)
- Analyzed the loss curves


### What I tried / what failed


### Technical details & thoughts
- understanding again that the models looks and refines the whole tomogram, the 30-50° region still gets filled /changed to some extent when using a 30° mask and a model trained on 50°. Explanation of the outcome in notes.md
- Loss analysis: (converge stable / no failed training / similar speed / training difficulty different absolute values)


## 2026-06-02

### What I did
- Training jobs train_mw40/50/60 completed on A100 via Slurm.
- Optimized the loss_curve analysis. Decided to use the lowest val_loss value as the point where I choose the model. Since fit_loss could overfit and was oscillating a lot. -> (999 at mw40, 989 at mw50 and mw60)
- run the refinement with the newly trained models. -> used halves.py again to end up with even and odd tomograms to calculate a resolution.
- notebook for FSC calculation and comparison for the three differently trained models.
- Interpretation and understanding of the FSC plots, which model performs best or what can we even say about the models from this experimental setup?
- Sub-Exp 2: Training angle matters, but less geometrically directly than inference angle. 60° training hurts consistency most. 40° training gives highest even/odd FSC, possibly because the training task is easier/more conservative. FSC-0.5 is never reached
- project update mail for manuel and yves.



### What I tried / what failed
- (training error) some subtomo .pt loading errors occurred near the end (Ran out of input / zip archive issue), but jobs completed and checkpoints were written. Need to verify refinement works with produced checkpoints.
- calculating the resolution for different setups/experiments, never hit the 0.5 FSC threshold -> no resolution calc. possible

### Technical details & thoughts
- What to do about comparing and calculating the resolution in voxel or angstrom size, when I never hit the 0.5 FSC threshold?
- Thoughts on which one to pursue:
The combined FSC comparison shows that both inference-time and training-time angle choices affect the reconstruction, mainly at high spatial frequencies. In Sub-Exp 1, changing the inference angle directly changes the Fourier mask used during refinement. Therefore, the observed differences are linked to geometric failure modes: too small angles under-fill the true missing wedge, while too large angles can over-mask and modify regions that should not be treated as missing. In Sub-Exp 2, the inference angle is fixed to 50°, so all models fill the same Fourier region. The differences there reflect the learned model behaviour rather than a change in reconstruction geometry.
Based on this, inference-time angle errors are the more physically direct and practically important error source to investigate further. Training-time errors affect high-frequency consistency, but a moderately wrong training angle can still produce reasonable results if the inference geometry is correct. Therefore, future work should focus on validating or estimating the correct inference-time missing-wedge angle and detecting underfilling/over-masking using Fourier-space diagnostics.



## 2026-06-03


### What I did
- reading ddw paper to find out about their ground truth used for the FSC plots and CC plots.
- understanding FSC calculation (which regions are used?)
- ddw does: FSC(synthetic ground truth, refined tomo)
- my code does: FSC(refined even, refined odd)
- fourier difference maps for input and refined tomos (30°, 50°, 70°)
- progress report / midterm report for the supervisors.

### What I tried / what failed
- computing the volume difference maps lead to a lot of jupyter notebook crashes.
- easy fix: choose planes first and only calculated the difference for each plane once (3 planes) instead of a whole volume.

### Technical details & thoughts
- Plot dimension such that it works in overleaf later
#Full width
figsize=(6.5,4)

#Wide 2-panel figure
figsize=(7.0,3.5)

#Square-ish figure
figsize=(5,4)
 

## 2026-06-04


### What I did
- training and fine tuning the parameters for my own model
- reducing parameters until the ra cluster GPU let me train my model without "out of memory" errors
- calculating fourier and real space difference maps for the subexp02
- using the difference maps as insightful results instead of FSC
- starting with drafts for methods and experiment section in report
- jupyter notebook for finalizing the plots

### What I tried / what failed
- getting the DDW paper layout for the combined axes plots of a sample
- somehow very hard to reconstruct

### Technical details & thoughts
- axes notation in plot (removed), should add all axis in the final plot s.t. same dimensions?
- from which and how to import the colormap/colorbars? Only one needed but provided from 3-6 pictures??




## 2026-06-08

### What I did
- cleaning up some older notebooks
- plotting report ready figures, deciding on which figures I think are smart to include
- Introduction and results draft text for report
- training a new parameter optimization model 
- conifg_02 in own model directory for specifications 

### What I tried / what failed
- FSC calculation between input and refined in older notebook, check with claude 
- try evaluation on FSC only in missing wedge? Why still so high? Function correct? Inputs correct?
- CUDA Out-of-memory depending on subtomo size and sliding window (GPU vram limited?)

ASK Wenxuan:
- How to get Figure 1: from official DDW Paper such that I can use it?
- Titles in report figures?
- FSC plots how to include in the report? 


### Technical details & thoughts
- check with wenxuan about the choosen plots
- dimensions of colormaps in the final comparison figures





## 2026-06-09

### What I did
- Introduction and Background drafts for report
- Notebook for calculating metrics corresponding to the difference maps
- such that one can compare not only visuals but also values

### What I tried / what failed
- FSC calculation of only the missing wedge region. Did bring FSC down to almost 0.5 Threshold for even/odd comparison. But that just shows me that for both even and odd, the model predicts roughly the same information for the fourier region, not really helpful as a comparison.

### Technical details & thoughts
- 



## 2026-06-10

### What I did
- repo clean up
- directory clean up
- rewriting notebooks because of structural changes
- readme
- report draft writing


### What I tried / what failed


### Technical details & thoughts



## 2026-06-12

### What I did
- changing the plot layouts for the final plots
- adding zoomed in regions for the most interesting part (physical wise)
- wrote a notebook for going through different slices (every 20 slices)
- and compare the difference maps to find slices with big differences
- then looked at this zoomed in slice regions and compared real space slices to spot differences / elongation
- possible to now compare zoomed in real space for subexp01 and subexp02 nevertheless outcome not as good
- as I hoped for


### What I tried / what failed
- training and refining my model had some errors, fixed it and restarted
- ra cluster problems or my .sh was wrong? not sure


### Technical details & thoughts
- my own trained model is not really looking like a huge difference to the tutorial model 
- config for the DDW trained model is on github, mine is config_03.yaml for report and comparison






## 2026-06-17

### What I did



### What I tried / what failed



### Technical details & thoughts
