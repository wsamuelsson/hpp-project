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
void ftgemm(floatType *A, floatType *B, floatType *C, int N);

void eye(floatType *A, int N);

int main(){

    int N;
    N = 1000;
    
    floatType *A = (floatType *)malloc(N*N*sizeof(floatType));
    floatType *L = (floatType *)malloc(N*N*sizeof(floatType));
    floatType *U = (floatType *)malloc(N*N*sizeof(floatType));
    floatType *A_org = (floatType *)malloc(N*N*sizeof(floatType));
    memset(&L[0], 0.0, N*N*sizeof(floatType));
    memset(&U[0], 0.0, N*N*sizeof(floatType));

    
    symmetric_random_matrix(&A[0], N);
    memcpy(A_org, A, N*N*sizeof(double));

    
    
    LU_factor(&A[0], N);
    
    get_L_U(&A[0], &L[0], &U[0], N);

    
    
    
    memset(&A[0], 0.0, N*N*sizeof(floatType));
    ftgemm(&L[0], &U[0], &A[0], N);
    

    floatType absSum = 0.0;
    for (int i = 0; i < N; i++)
    {
        absSum += abs(A[i] - A_org[i]);
    }
    printf("Abssum=%lf\n", absSum);
    
    
    free(A);
    free(U);
    free(L);
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
            
            printf("%lf  ", A[j*N + i]);
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
           
            A[j*N + i] = random_num;
            A[i*N + j] = random_num;
        }
    }
    
}

void LU_factor(floatType *A,  int N){
    int k,i,j;
    floatType Akk_inv;
    for(k=0;k<N;k++){
        Akk_inv = 1.0/A[k*N + k];
        for(i=k+1;i<N;i++){
            A[k*N+i] *= Akk_inv; 
        }
        
        for(i=k+1;i<N;i++){
            for(j=k+1;j<N;j++){
                A[j*N + i] -= A[k*N + i]*A[j*N + k];
            }
        }
    }
}


void ftgemm(floatType *A, floatType *B, floatType *C, int N){
    //This is inspired by https://netlib.org/lapack/explore-html/dd/d09/group__gemm_ga1e899f8453bcbfde78e91a86a2dab984.html
    //Computes: C = C + A*B

    int i,j,k;
    floatType temp;

    for(j=0;j<N;j++){
        for(k=0;k<N;k++){
            temp = B[j*N+k];
            for(i=0;i<N;i++)
                C[j*N+i] += A[k*N+i] * temp; 
        }
    }

}

void get_L_U(floatType *LU, floatType *L, floatType* U, int N){
    int i,j;

    for(i=0;i<N;i++){
        L[i*N + i] = 1.0;
        U[i*N + i] = LU[i*N + i];
        for(j=i+1;j<N;j++){
            U[N*j + i] = LU[N*j + i];
            L[i*N + j] = LU[i*N + j];
        }
    }

}