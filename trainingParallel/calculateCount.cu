__global__ void calculateCount(int *keypoints ,const unsigned char  *in, int *allProbablities,  int *allIndexList, int patchSize, int width, int height, int fernNum, int fernSize){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    //int index2 = index*2+1;
    //printf("%d %d\n", allIndexList[index*2], allIndexList[index2]);

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

   int patchHeight = endX - startX;
  

   int patch[1024];


    int count = 0;
    for(int j= 0; j < patchHeight; j++){
        for(int i = startY ; i < endY; i++){
            patch[count] = in[startX*height+i];
            count++;
        }
        startX = startX +1;
    }
    /*count = 0;
    int decimalNum = 0;
    for(int i = 0; i< fernNum ; i++){
        for(int j = 0; j < fernSize; j++){
                
        }
    }*/
   

}