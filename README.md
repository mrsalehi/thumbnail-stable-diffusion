# thumbnail-stable-diffusion

I fine tuned stable diffusion on youtube thumbnails and the video title. I was hoping to create a tool that help Youtubers brain storm with diffusion models to create eye-catchy thumbnails. The results for kids channels and cartoonish thumbnails were particulary cool. However, I found that stable diffusion had a hard time generating photo-realistic thumbnails.



Here are some of the generated thumbnails with the fine tuned model. First row of all the figures are the generated images by stable diffusion (not fine-tuned) for the video title. Rest of the rows are outputs of the fine tuned model on the data of the channel:


Title: animal figurines from fruits:
<p align="center">
<img src="assets/stable_diffusion_generated_youtube_thumbnails/animal figurines from fruits_step_0_5000_1.png" alt="drawing" width="800"/>
</p>


Title: baby in a space suit:
<p align="center">
<img src="assets/stable_diffusion_generated_youtube_thumbnails/baby in a space suit_step_0_5000_1.png" alt="drawing" width="800"/>
</p>


Title: baby is in a spaceship:
<p align="center">
<img src="assets/stable_diffusion_generated_youtube_thumbnails/baby is in a spaceship_step_0_5000_1.png" alt="drawing" width="800"/>
</p>


Title: blowing candles on birthday cake:
<p align="center">
<img src="assets/stable_diffusion_generated_youtube_thumbnails/blowing candles on birthday cake_step_0_5000_1.png" alt="drawing" width="800"/>
</p>

Title: playing with baloons:
<p align="center">
<img src="assets/stable_diffusion_generated_youtube_thumbnails/playing with baloons_step_0_5000_1.png" alt="drawing" width="800"/>
</p>

Title: playing with other kids:
<p align="center">
<img src="assets/stable_diffusion_generated_youtube_thumbnails/playing with other kids_step_0_5000_1.png" alt="drawing" width="800"/>
</p>

Title: running in a green garden:
<p align="center">
<img src="assets/stable_diffusion_generated_youtube_thumbnails/running in a green garden_step_0_5000_1.png" alt="drawing" width="800"/>
</p>

Title: sad cocomelon baby:
<p align="center">
<img src="assets/stable_diffusion_generated_youtube_thumbnails/sad cocomelon baby_step_0_5000_1.png" alt="drawing" width="800"/>
</p>

Title: swimming in the pool:
<p align="center">
<img src="assets/stable_diffusion_generated_youtube_thumbnails/swimming in the pool_step_0_5000_1.png" alt="drawing" width="800"/>
</p>

Title: Birthday at the farm song with cocomelon:
<p align="center">
<img src="assets/Birthday at the farm song with cocomelon_step_first_good_sample.png" alt="drawing" width="800"/>
</p>


Another interesting observation that I had was that when stable diffusion is being fine-tuned, the loss usually does not decrease fast and when it does decrease, perhaps it is overfitting to the data. Here is a figure that demonstrates the progress of generated images throughout fine-tuning: 

Title: Birthday at the farm song with cocomelon:
<p align="center">
<img src="assets/Birthday at the farm song with cocomelon_step_0_50000_2.png" alt="drawing" width="800"/>
</p>

First row is without fine tuning and next rows are when fine tuning progresses. As seen above, images in row 3 and after are identical to the images in training data for the exact same title.





