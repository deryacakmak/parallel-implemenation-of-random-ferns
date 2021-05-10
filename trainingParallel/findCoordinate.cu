__global__ void findCoordinate(float *A, int *keypoints,  int *newKeypoints, int lenght){

    int index = blockIdx.x * blockDim.x + threadIdx.x;


    if(index < lenght){

        float a00 = A[0];
        float a01 = A[1];
        float a10 = A[3];
        float a11 = A[4];
        float t0 = A[2];
        float t1 = A[5];      
       
       int x = keypoints[index];
        int y = keypoints[index+lenght];

  

        float xp = a00*y + a01*x + t0;
        float yp = a10*y +a11*x + t1;


       int x1 = (int)xp;
        int y1 = (int)yp;

        //printf("%d %d %f %f %d %d %f %f %f %f %f %f %f\n",x1, y1, xp, yp, x, y, a00,a01,a10,a11,t0,t1);
        

       int index2 = index*2+1;
       int index1 = index*2;
    
        newKeypoints[index1] = x1;
        newKeypoints[index2] = y1;
    }

   


}