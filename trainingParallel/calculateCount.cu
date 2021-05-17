__global__ void calculateCount(int *keypoints ,const unsigned char  *in, int *allProbablities,  int *allIndexList, int patchSize, int width, int height, int fernNum, int fernSize, int lenght){

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

   int patchHeight = endX - startX;
   int patchLenght = patchHeight * (endY - startY);

   int patch[1024];


    int count = 0;
    for(int j= 0; j < patchHeight; j++){
        for(int i = startY ; i < endY; i++){
            patch[count] = in[startX*height+i];
            count++;
        }
        startX = startX +1;
    }

    int I1, I2,num, decimalNum;
    for(int i = 0; i< fernNum ; i++){
        decimalNum = 0;
        num = 1;
        for(int j = 0; j < fernSize; j++){
             I1 = allIndexList[fernSize*i*2+(j*2)];
             I2 = allIndexList[fernSize*i*2+(j*2)+1];
            if(I1 <  patchLenght && I2 < patchLenght){
                if(patch[I1] < patch[I2]){
                    decimalNum = decimalNum +num;
                }
                num = num *2;
            }     
        }
        allProbablities[index*lenght+decimalNum] = allProbablities[index*lenght+decimalNum]+ 1;  
    }
   

}