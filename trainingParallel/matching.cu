__global__ void matching(int *keypoints ,const unsigned char  *in, int *allProbablities, int *allIndexList,  int *matchingResult , int width, int height, int lenght, int fernNum, int fernSize, int patchLenght){

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int patchSize =(int)(patchLenght /2);
    int x = keypoints[index*2];
    int y = keypoints[index*2+1];

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

   int patchHeight = endY - startY;
   int patcWidth = endX - endY;
   int size = patchHeight*patcWidth;
   
    int patch[1024];


    int count = 0;
    for(int j= 0; j < patchHeight; j++){
        for(int i = startY ; i < endY; i++){
            patch[count] = in[startX*height+i];
            count++;
        }
        startX = startX +1;
    }

    
    int result[250];

    int I1, I2,num, decimalNum, index2;
    for(int i = 0; i< fernNum ; i++){
        decimalNum = 0;
        num = lenght/2;
        for(int j = 0; j < fernSize; j++){
             index2 = (fernSize*i*2)+(j*2);
             I1 = allIndexList[index2];
             I2 = allIndexList[index2+1];
            if(I1 <  size && I2 < size){
                if(patch[I1] < patch[I2]){
                    decimalNum = decimalNum +num;
                }
                num = num /2; 
                
            } 
             
        }
        for(int j = 0; j< 250; j++){
            result[j] = result[j] + logf(allProbablities[j*lenght+decimalNum]);
        }

    }

    num = result[0];
    index2 = 0;
    for(int k = 1; k < 250; k++){
        decimalNum = result[k];
        if( decimalNum> num ){
            num = decimalNum;
            index2 = k;
        }
    }

  matchingResult[index] = index2;


}