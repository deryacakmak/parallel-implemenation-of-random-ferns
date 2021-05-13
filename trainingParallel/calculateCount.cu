__global__ void calculateCount(int *keypoints ,const unsigned char  *in, unsigned char *out, int patchSize, int width, int height){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int y = keypoints[index*2];
    int x = keypoints[index*2+1];

    int startX = x - patchSize;
    int endX = x + patchSize;

    int startY = y - patchSize;
    int endY = y + patchSize;


    if(startX < 0  ){
            startX = 0;
        }
        
        
    if (endX >= width ){
         endX = width -1;
    }
       
        
    if(startY < 0 ){
         startY = 0;
    }
       
        
    if (endY >= height){
        endY = height -1;
    }


    
   // int out[end - start];

   int patchHeight = endX - startX;



    int count = 0;
    for(int j= 0; j < patchHeight; j++){
        for(int i = startY ; i < endY; i++){
            out[count] = in[startX*height+i];
            count++;
        }
        startX = startX +1;
    }
 


        
    

    //printf("%d**", in[y*width+x]);

}