The process of obtaining a new image was performed by applying an affine transformation matrix randomly generated to the image given in this section.

Make sure you have set up OpenCV and NumPy for use in serial and parallel implementation, and CUDA for use in parallel implementation.

Sample images you can use are given in the dataset folder. The outputs obtained at the end of the program are saved in the output file.

You can use the following command to run the affineDeformation.py file:
        
        python3 affineDeformation.py -f fileName.xxx

Note: fileName.xxx is the input image for which you want to apply the affine transformation
process.

When you run it without any problems, you should get an output as follows:


![Screenshot from 2021-03-01 20-42-26](https://user-images.githubusercontent.com/36774966/109540895-29ed3900-7ad4-11eb-81a4-1e15038eb547.png)

You can use the following command to run the affineDeformationParallel.py file:
        
        sudo nvprof python3 affineDeformationParallel.py -f fileName.xxx

Note: In this part, If you get an error regarding the installation of pycuda while running this  command, you can run the following command for installation:

        sudo pip install pycuda
        

When you run it without any problems, you should get an output as follows:

![Screenshot from 2021-03-01 20-40-35](https://user-images.githubusercontent.com/36774966/109541028-57d27d80-7ad4-11eb-8b06-30acb830c355.png)

You can also use --print-gpu-trace. You can use the following command by running:

      sudo nvprof --print-gpu-trace python3 affineDeformationParallel.py -f fileName.xxx

When you run it without any problems, you should get an output as follows:

![Screenshot from 2021-03-01 21-12-18](https://user-images.githubusercontent.com/36774966/109541067-6751c680-7ad4-11eb-90ba-d0632d522abf.png)

If you do not want a detailed output like the one above, you can run the following command:

        python3 affineDeformationParallel.py -f fileName.xxx

