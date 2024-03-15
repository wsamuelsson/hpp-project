#include<stdio.h>
#include<omp.h>
#include<string.h>
#include<stdlib.h>
#include<omp.h>
#include<assert.h>
#include<math.h>

typedef double floatType;

void random_matrix(floatType *A,   int N);
void print_matrix(floatType *A,   int N);
void symmetric_random_matrix(floatType *A,  int N);
void block_matrix(floatType *A, floatType *A_block,  int N, int block_sz);
void LU_factor(floatType *A,  int N);
void get_L_U(floatType *LU, floatType *L, floatType* U, int N);
void get_L_U_parallel(floatType *LU, floatType *L, floatType *U, int N);
void ftgemm(floatType *A, floatType *B, floatType *C, int N);
void LU_factor_parallel(floatType *A,  int N, int nThreads);
void inline transpose_matrix(floatType *A, floatType *AT, int N);
void eye(floatType *A, int N);

int main(int argc,  char**argv){

    if(argc != 3){
        printf("Usage: ./matrix N nThreads\n");
        return -1;
    }
    int N = atoi(argv[1]);
    int nThreads = atoi(argv[2]);
    int nMax = 4096;
    
    floatType *A = (floatType *)malloc(N*N*sizeof(floatType));
    floatType *L = (floatType *)malloc(N*N*sizeof(floatType));
    floatType *U = (floatType *)malloc(N*N*sizeof(floatType));
    floatType *A_org = (floatType *)malloc(N*N*sizeof(floatType));
    floatType *A_T = (floatType *)malloc(N*N*sizeof(floatType));
    
    memset(&L[0], 0.0, N*N*sizeof(floatType));
    memset(&U[0], 0.0, N*N*sizeof(floatType));

    
    random_matrix(&A[0], N);
    memcpy(A_org, A, N*N*sizeof(double));
    //print_matrix(&A[0], N);
    //printf("\n");
    
    transpose_matrix(&A[0], &A_T[0], N); //Lu factor parallel expects column major matrix
    double t0 = omp_get_wtime();
    LU_factor_parallel(&A_T[0], N, nThreads); //Perform on column major matrix
    printf("LU (parallel) for N=%d took %lf seconds\n", N, omp_get_wtime() - t0);
    //transpose_matrix(&A_T[0], &A[0], N); //Result is in column major format -> Transpose back to get row major
    get_L_U(&A_T[0], &L[0], &U[0], N);
    //print_matrix(&A_T[0], N);
    printf("\n");

    memset(&A[0], 0.0, N*N*sizeof(floatType));
    //ftgemm(&L[0], &U[0], &A[0], N);
    

    floatType absSum = 0.0;
    for (int i = 0; i < N*N; i++)
    {
        absSum += abs(A[i] - A_org[i]);
    }
    printf("Abssum=%.12lf\n\n", absSum);
    
    
    memset(&L[0], 0.0, N*N*sizeof(floatType));
    memset(&U[0], 0.0, N*N*sizeof(floatType));
    memcpy(A, A_org, N*N*sizeof(double));
    #pragma omp flush
    t0 = omp_get_wtime();
    LU_factor(&A[0], N);
    printf("LU (Serial) for N=%d took %lf seconds\n", N, omp_get_wtime() - t0);
    //print_matrix(&A[0], N);
    get_L_U(&A[0], &L[0], &U[0], N);
    
    memset(&A[0], 0.0, N*N*sizeof(floatType));
    //ftgemm(&L[0], &U[0], &A[0], N);
    

    //print_matrix(&A[0], N);


    
    absSum = 0.0;
    for (int i = 0; i < N*N; i++)
    {
        absSum += abs(A[i] - A_org[i]);
    }
    printf("Abssum=%.12lf\n", absSum);
    
    
    
    free(A);
    free(U);
    free(L);
    free(A_org);
    return 0;

}  


void random_matrix(floatType *A,  int N){
     int i,j;
    
    for(i=0;i<N;i++){    
        for(j=0; j<N;j++){
            A[j*N + i] = (floatType) 1000 * rand()/RAND_MAX;
        }
    }


}

void print_matrix(floatType *A,  int N){
     int i,j;
    for(i=0;i<N;i++){
        for(j=0; j<N;j++){
            
            printf("%lf  ", A[i*N + j]);
        }
    printf("\n");
    }
    
}

void symmetric_random_matrix(floatType *A,  int N){
    floatType random_num;
     int i, j;
    for (i=0;i<N;i++){
        for (j=i;j<N;j++){
            random_num = (floatType) (rand()%10 + 1);
            A[i*N + j] = random_num;
            A[j*N + i] = random_num;
        }
    }
    
}

void LU_factor(floatType *A,  int N){
    int k,i,j;
    floatType Akk_inv;
         
        for(k=0;k<N;k++){
            Akk_inv = 1.0/A[k*N + k];
           
            for(i=k+1;i<N;i++){
                A[i*N+k] *= Akk_inv; 
            }
            
            //printf("k=%d\n", k);
            for(i=k+1;i<N;i++){
                
                //printf("i=%d\n", i);
                for(j=k+1;j<N;j++){
                    //printf("i*N + j=%d\n", i*N+j);
                    A[i*N + j] -= A[i*N + k]*A[k*N + j];
                }
            }
        }
    
    
}


void ftgemm(floatType *A, floatType *B, floatType *C, int N){
    //This is inspired by https://netlib.org/lapack/explore-html/dd/d09/group__gemm_ga1e899f8453bcbfde78e91a86a2dab984.html
    //Computes: C = C + A*B

    int i,j,l;
    floatType temp;

    for(j = 0; j < N; j++) {
        for(i = 0; i < N; i++) {
            temp = 0.0;
            for(l = 0; l < N; l++)
                temp += A[j*N + l] * B[l*N + i];
            C[j*N + i] = temp;
        }
}

}

void get_L_U(floatType *LU, floatType *L, floatType* U, int N){
    int i,j;

    for(i=0;i<N;i++){
        L[i*N + i] = 1.0;
        U[i*N + i] = LU[i*N + i];
        for(j=i+1;j<N;j++){
            U[i*N + j] = LU[N*i + j];
            L[j*N + i] = LU[j*N + i];
        }
    }

}

void get_L_U_parallel(floatType *LU, floatType *L, floatType* U, int N){
    int i,j;
    for(i=0;i<N;i++){
        L[i*N + i] = 1.0;
        U[i*N + i] = LU[i*N + i];
        for(j=i+1;j<N;j++){
            U[i*N + j] = LU[N*j + i];
            L[j*N + i] = LU[i*N + j];
        }
    }
}

void LU_factor_parallel(floatType *A,  int N, int nThreads){
    if(nThreads >= N){
        printf("Dont use this many cores!\n");
        return;
    }
    int k,i,j, nlim;

    floatType Akk_inv;
    omp_lock_t *locks = (omp_lock_t *)malloc(N * sizeof(omp_lock_t)); //One lock per column
    
    //Init locks
    for(k=0;k<N;k++){
        
        omp_init_lock(&locks[k]);
        
    }
    
    int col ,threadID, start, k_temp, row, a;
    int n_break_point = 1000;

    nlim = N - n_break_point;
    if(N < n_break_point)
        nlim = N; 
    
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(nThreads)  private(col, k, start, threadID, Akk_inv, k_temp, row, i, a) 
    {   
        
        
        threadID = omp_get_thread_num();
        
        
        for(col=threadID;col<N;col+=nThreads){
            omp_set_lock(&locks[col]);
        }
                
        #pragma omp barrier

        if(threadID == 0){ //Master thread computes
            //printf("Normalizing column=0\n");
            Akk_inv = 1.0/A[0];
            for(i=1;i<N;i++){
                
                A[i] *= Akk_inv;
            }
            omp_unset_lock(&locks[0]);
            
        }
        
        
        for(k=0;k<nlim;k++){
            
            omp_set_lock(&locks[k]);
            omp_unset_lock(&locks[k]);
            
            start = (k/nThreads)*nThreads; //Integer div
            if(start+threadID <= k){
                start+=nThreads;
            }
            
            
            for(col=start+threadID;col<N;col+=nThreads){
                
                for(row=k+1;row<N;row++){
                    A[col*N + row] -= A[k*N+row]*A[N*col+k];
                }
                
                if(col == k+1 && col < nlim){
                    k_temp = k+1;
                    Akk_inv = 1.0/A[k_temp*N + k_temp];
                    
                    for(i=k_temp+1;i<N;i++){
                        A[k_temp*N + i] *= Akk_inv;
                    
                    }
                    
                    
                    omp_unset_lock(&locks[k_temp]);
                }
            }
        }
        
        
    }
       
            
    //Destroy locks
    for(k=0;k<N;k++){
        omp_destroy_lock(&locks[k]);
        }
    printf("Parallel LU took %lf seconds\n", omp_get_wtime() - t0);
    printf("We have now elimated %d columns\n", nlim);
    floatType *A_T = (floatType *)malloc(sizeof(floatType)*N*N);
    t0 = omp_get_wtime();
    transpose_matrix(&A[0], &A_T[0], N);
    printf("Transpose inside took: %lf\n", omp_get_wtime() - t0);

    t0 = omp_get_wtime();
    for(k = nlim;k<N;k++){
        Akk_inv = 1.0/A_T[k*N + k];
        for(i=k+1;i<N;i++){
            A_T[i*N + k] *= Akk_inv;
        }
        for(i=k+1;i<N;i++){
            for(j=k+1;j<N;j++){
                A_T[i*N + j] -= A_T[i*N + k]*A_T[k*N + j];
            }
    }
    }
    printf("Serial took %lf seconds: \n", omp_get_wtime() - t0);

    
    memcpy(&A[0], &A_T[0], N*N*sizeof(floatType));
    
    free(A_T);
    
        
    free(locks);
    
}

void transpose_matrix(floatType *A, floatType *AT, int N){
    int i,j;
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            AT[j*N + i] = A[i*N+j];
        }
    }
}