
__global__ void affineDeformation(float *A, int *in, int *out, int newWidth, int newHeight ,int oldWidth, int oldHeight)
  {
    
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
 	

    
    if(Col < newWidth && Row < newHeight){
            
        float a00 = A[0];
        float a01 = A[1];
        float a10 = A[3];
        float a11 = A[4];
        float t0 = A[2];
        float t1 = A[5];
        float det_A = a00*a11 - a01*a10;
        
        float inv_det_A = 1.0 / det_A;
        
        int stride = blockDim.x * gridDim.x* blockDim.y * gridDim.y * blockDim.z * gridDim.z;
        
        int currentPixel = Row * newWidth + Col;
        
        
        while( currentPixel< stride)
        {
         
	        float xa = Col - t0;
               float ya = Row - t1;
               float xp = inv_det_A * (a11 * xa - a01 * ya);
               float yp = inv_det_A * (a00 * ya - a10 * xa);
               
               int value = 0;
                
               int xi = (int)xp;
		int yi = (int)yp;
		if (xi >= 0 && yi >= 0 && xi < oldWidth-1 && yi < oldHeight-1){
            
                int I00 = in[yi * oldWidth + xi];
                int I10 = in[yi * oldWidth + (xi+1)];
                int I01 = in[(yi + 1) * oldWidth + xi];
                int I11 = in[(yi + 1) * oldWidth + (xi+1)];
                float alpha = xp - xi;
                float beta = yp - yi;
                float interp = (1.0 - alpha) * (1.0 - beta) * I00
                        + (1.0 - alpha) * beta * I01
                        + alpha * (1.0 - beta) * I10
                        + alpha * beta * I11;
                int interpi = (int)interp;
                if (interpi < 0)
                        interpi = 0;
                else if (interpi > 255)
                        interpi = 255;
                value = interpi;
            
            }
            
  

        	out[currentPixel] = value;
        
         currentPixel += stride;
        }
        
            
            }    

  }
  



