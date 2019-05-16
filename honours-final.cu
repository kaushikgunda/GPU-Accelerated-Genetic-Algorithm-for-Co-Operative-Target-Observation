#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<bits/stdc++.h>
#include <cuda_runtime.h>
#include<cuda.h>
#include <curand_kernel.h>
#include<curand.h>
#include <sys/time.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/extrema.h>
using namespace std;
#define NUM_THREADS 256
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

	__host__ __device__
unsigned int hash(unsigned int a)
{
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	return a;
}

struct RandomNumberFunctor : 
	public thrust::unary_function<unsigned int, float>
{
	unsigned int mainSeed;
	RandomNumberFunctor(unsigned int _mainSeed) : 
		mainSeed(_mainSeed) {}

	__host__ __device__
		float operator()(unsigned int threadIdx) 
		{
			unsigned int seed = hash(threadIdx) * mainSeed;

			// seed a random number generator
			thrust::default_random_engine rng(seed);

			// create a mapping from random numbers to [0,1)
			thrust::uniform_real_distribution<float> u01(0,1);

			return u01(rng);
		}
};



void initialize_genes(thrust::host_vector<int> &genes_x,thrust::host_vector<int> &genes_y,int m, int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			genes_x[i*m+j]=rand()%3-1;
			genes_y[i*m+j]=rand()%3-1;
		}
	}
}
__global__ void compute_fitness(float* fitness,int* genes_x,int* genes_y,int* observers_x,int* observers_y,int* targets_x,int* targets_y,int *observer_count,int* target_count,int *radarRange){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;	
	if (tid >= 256 ) 
	{
		return;
	}
	//printf("Radar range %d\n",*radarRange);
	int m=*(observer_count);
	for(int i=0;i<m;i++)
		if(!((observers_x[i]+genes_x[tid*m+i])>=0 && (observers_x[i]+genes_x[tid*m+i])<=150 && (observers_y[i]+genes_y[tid*m+i])>=0 && (observers_y[i]+genes_y[tid*m+i])<=150))
		{fitness[tid]=-100;return;}
	bool observed[27];
	for(int i=0;i<(*target_count);i++)observed[i]=0;
	//if(tid==0)
	//printf("Fitness for chromosome 0: %d\n",fitness[0]);

	for(int i=0;i<(*target_count);i++){
		for(int j=0;j<m;j++){
			int dist = (observers_x[j]+genes_x[tid*m+j]-targets_x[i])*(observers_x[j]+genes_x[tid*m+j]-targets_x[i])+(observers_y[j]+genes_y[tid*m+j]-targets_y[i])*(observers_y[j]+genes_y[tid*m+j]-targets_y[i]);
		//if(tid==0)
		//printf("Target %d,Observer %d,dist %f\n",i,j,sqrtf(dist));
			if(sqrtf(dist)<=(*radarRange)){
				//printf("yoyo\n");
				observed[i]=1;
			}
		}
	}
	
	for(int i=0;i<(*target_count);i++)
		if(observed[i])
		fitness[tid]+=1;
	//if(tid==0)
        //printf("Fitness for chromosome 0: %d\n",fitness[0]);
}
__global__ void selection(float* fitness,int* selections,float* random,int* cross,int n ){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int population_size = 256;
	if(tid>=population_size){
		return ;
	}
	int vs = random[tid] * population_size;
	int vs2 = random[tid+population] * population_size;
	selections[tid] = fitness[tid] > fitness[vs] ? tid :vs;
	selections[tid] = fitness[selections[tid]]> fitness[vs2] ? selections[tid] : vs2;
	//printf("%d %d\n",tid,selections[tid]);
	// CrossOver points Kernel
	if(tid%2==1)return;
	float r1 = random[tid+population_size*2];
	if(r1>=0.05){
		cross[tid]=random[tid+3*population_size]*(n/population_size);
	}
	else{
		cross[tid]=0;
		cross[tid+1]=0;
	}
}
__global__ void crossover(int* genes_x,int* genes_y,int* offspring_x,int* offspring_y,int *selections,int m,int n,int* cross){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tid2 = threadIdx.y;
	if(tid%2==1)return;
	if(tid>=n) return;
	if( tid2 <= cross[tid] ){
		offspring_x[(tid+0)*m+tid2 ] = genes_x[selections[tid+0]*m + tid2 ];
		offspring_x[(tid+1)*m+tid2 ] = genes_x[selections[tid+1]*m + tid2 ];
		offspring_y[(tid+0)*m+tid2 ] = genes_y[selections[tid+0]*m + tid2 ];
		offspring_y[(tid+1)*m+tid2 ] = genes_y[selections[tid+1]*m + tid2 ];
	}else{
		offspring_x[(tid+1)*m+tid2 ] = genes_x[selections[tid+0]*m + tid2 ];
		offspring_x[(tid+0)*m+tid2 ] = genes_x[selections[tid+1]*m + tid2 ];
		offspring_y[(tid+1)*m+tid2 ] = genes_y[selections[tid+0]*m + tid2 ];
		offspring_y[(tid+0)*m+tid2 ] = genes_y[selections[tid+1]*m + tid2 ];
	}	
}	
__global__ void mutation(int* offspring_x,int* offspring_y, float* random,int m,int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tid2 = threadIdx.y;
	if(tid>=m)return;
	if( random[tid] >=0.95){
		if(random[m*n+tid*n+tid2]<=0.33)
			offspring_x[tid*n + tid2]=-1;
		else if(random[m*n+tid*n+tid2]>=0.67)
			offspring_x[tid*n + tid2]=0;
		else
			offspring_x[tid*n + tid2]=1;

		if(random[2*m*n+tid*n+tid2]<=0.33)
			offspring_y[tid*n + tid2]=-1;
		else if(random[2*m*n+tid*n+tid2]>=0.67)
			offspring_y[tid*n + tid2]=0;
		else
			offspring_y[tid*n + tid2]=1;
	}
}
int randint(int a,int b){
	return rand()%3-1;
}
int main(int argc,char** argv){
	float score = 0;
	srand (time(NULL));
	unsigned int seed = time(NULL);
	
	int n=12;//Number of observers
	int m=24;//Number of targets
	int radius=strtol(argv[1],NULL,10);//=25;//Radius of observation
 	//cout << radius << endl;
	int t =strtol(argv[2],NULL,10);
	int chromo_size = n;
	if(argc>3){n=strtol(argv[3],NULL,10);m=strtol(argv[4],NULL,10);}
	int population_size = 256;	
	//Cuda Memory allocation
	int *observers_x,*observers_y,*targets_x,*targets_y,*observers_count,*targets_count,*radar_range;
	int host_observers_x[n],host_observers_y[n],host_targets_x[m],host_targets_y[m];	
	cudaMalloc((void**)&observers_x,n*sizeof(int)); cudaMalloc((void**)&observers_y,n*sizeof(int));
	cudaMalloc((void**)&observers_count,sizeof(int));  cudaMalloc((void**)&targets_x,m*sizeof(int));
	cudaMalloc((void**)&targets_y,m*sizeof(int));   cudaMalloc((void**)&targets_count,sizeof(int));
	cudaMalloc((void**)&radar_range,sizeof(int));
	//Observer and target position allocation
	for(int a=0;a<n;a++){host_observers_x[a]=rand()%150;host_observers_y[a]=rand()%150;}
	for(int b=0;b<m;b++){host_targets_x[b]=rand()%150;host_targets_y[b]=rand()%150;}


	thrust::host_vector<int> genes_x(chromo_size * population_size);
	thrust::host_vector<int> genes_y(chromo_size * population_size);
	thrust::host_vector<int> offspring_x(chromo_size * population_size);
	thrust::host_vector<int> offspring_y(chromo_size * population_size);
	thrust::host_vector<float> fitness(population_size);
	thrust::host_vector<float> random(4 * population_size);
	thrust::host_vector<int> selections(population_size);
	thrust::device_vector<int> d_genes_x;
	thrust::device_vector<int> d_genes_y;
	thrust::device_vector<int> d_offspring_x;
	thrust::device_vector<int> d_offspring_y;
	thrust::device_vector<int> d_reverse_index(chromo_size * population_size * 2);
	thrust::device_vector<float> d_fitness(population_size,0);
	thrust::device_vector<int> crossover_points(population_size,0);
	thrust::device_vector<float> d_random(4 * population_size);
	thrust::device_vector<float> d_random2(3 * population_size*chromo_size);
	thrust::device_vector<int> d_selections(population_size);

	int target_dpick_x[m],target_dpick_y[m];
	for(int a=0;a<m;a++){target_dpick_x[a]=randint(-1,1);target_dpick_y[a]=randint(-1,1);}
	for(int tim=0;tim<1500;tim++){
		//printf("Time step: %d\n",tim);
		// Test -0 Initial observers and targets positions
		/*
		cout << "Observer Positions\n" ;
		for(int a=0;a<n;a++){cout << "Observer " << a << ": " << host_observers_x[a] << " " << host_observers_y[a] << endl;}
		cout << endl;
		cout << "Target Positions \n";
		for(int a=0;a<m;a++){cout << "Target " << a << ": " << host_targets_x[a] << " " << host_targets_y[a] << endl;}
		cout << endl;
		*/
		//Memory copy  from host to device
		cudaMemcpy(observers_count,&n, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(targets_count,  &m, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(observers_x,host_observers_x,n*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(observers_y,host_observers_y,n*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(targets_x,host_targets_x,m*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(targets_y,host_targets_y,m*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(radar_range,&radius,sizeof(int),cudaMemcpyHostToDevice);
		int blks = (population_size + NUM_THREADS ) / (NUM_THREADS);
		initialize_genes(genes_x, genes_y, chromo_size, population_size);
		d_genes_x=genes_x;
		d_genes_y=genes_y;
		d_offspring_x = d_genes_x;
		d_offspring_y = d_genes_y;
		float* d_fitness_raw = thrust::raw_pointer_cast(&d_fitness[0]);
		int* d_genes_raw_x = thrust::raw_pointer_cast(&d_genes_x[0]);
		int* d_genes_raw_y = thrust::raw_pointer_cast(&d_genes_y[0]);
		for(int iter=0;iter<1;iter++){
			//printf("Generation: %d\n",iter);
			// Test -1 Printing Genes 
			/*
			printf("Chromosome 1\n");
			for(int a=0;a<n;a++){
				cout << "Gene " << a << ": " << genes_x[a] << " "<<genes_y[a] << endl;
			}
			cout << endl;
			printf("Chromosome 2\n");
                        for(int a=n;a<2*n;a++){
                                cout << "Gene " << a-n << ": " << genes_x[a] << " "<<genes_y[a] << endl;
                        }
			cout << endl;
			*/
		
			compute_fitness<<<blks,NUM_THREADS>>>(d_fitness_raw,d_genes_raw_x,d_genes_raw_y,observers_x,observers_y,targets_x,targets_y,observers_count,targets_count,radar_range);
			//fitness = d_fitness;
			thrust::copy(d_fitness.begin(), d_fitness.end(), fitness.begin());
			//Test -2 Printing Fitness
			/*for(int a=0;a<2;a++)
			cout << "Fitness for chromosome " <<  a << ": " << fitness[a] << endl;
			*/
			
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );

			// Normalize fitness (sum of fitness = 1)
			//float total_fitness = thrust::reduce(d_fitness.begin(), d_fitness.end());
			//thrust::constant_iterator<float> normalization(total_fitness);
			//thrust::transform(d_fitness.begin(), d_fitness.end(), normalization, d_fitness.begin(), thrust::divides<float>());

			// Selection
			int* d_offspring_x_raw = thrust::raw_pointer_cast(&d_offspring_x[0]);
			int* d_offspring_y_raw = thrust::raw_pointer_cast(&d_offspring_y[0]);
			float* d_random_raw = thrust::raw_pointer_cast(&d_random[0]);
			int* d_selections_raw = thrust::raw_pointer_cast(&d_selections[0]);
			int* crossover_points_raw = thrust::raw_pointer_cast(&crossover_points[0]);
			// Generate 4n random numbers
			thrust::transform(thrust::counting_iterator<int>(0),
					thrust::counting_iterator<int>(4 * population_size),
					d_random.begin(), 
					RandomNumberFunctor(seed));

			// Uses 2*n random numbers for selection and 2n random numbers for crossoverpoint allocation
			// Selection + CrossOver Point selection
			selection<<<blks, NUM_THREADS >>> ( d_fitness_raw,d_selections_raw,d_random_raw,crossover_points_raw,d_genes_x.size());
			
			// Test-3 Printing Selection result
			thrust::copy(d_selections.begin(), d_selections.end(), selections.begin());
			//for(int a=0;a<2;a++){ cout << "Selection for chrosome " << a << "is " << selections[a] << endl;}	
			// Test -4 Printing cross over points
			/*thrust::host_vector<int> host_crossovers(population_size,0);
			thrust::copy(crossover_points.begin(),crossover_points.end(),host_crossovers.begin());
			for(int a=0;a<2;a++){ cout << "CrossOver Point for chrosome " << a << "is " << host_crossovers[a] << endl;}
			*/
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );

	
			//d_random_raw = thrust::raw_pointer_cast(&d_random[population_size]);  
			//Test - 5 Printing genes for two pairs of selection results and their corresponding cross over points
			/*cout << "Genes for selection for first 2 chrosomes\n";
			cout << "Parent 1 for chromosome 0 " << endl;
			for(int a=0;a<n;a++){
                                cout << "Gene " << a << ": " << genes_x[a] << " "<<genes_y[a] << endl;
                        }
			cout << "Parent 2 for chromosome 0 " << selections[0] << endl;
			for(int a=0;a<n;a++){
                                cout << "Gene " << a << ": " << genes_x[selections[0]*n+a] << " "<<genes_y[selections[0]*n+a] << endl;
                        }
			cout << endl << endl;
			cout << "Parent 1 for chromosome 2 " << endl;
			
			 for(int a=2*n;a<3*n;a++){
                                cout << "Gene " << a << ": " << genes_x[a] << " "<<genes_y[a] << endl;
                        }
                        cout << "Parent 2 for chromosome 2 " << selections[2] << endl;
			 for(int a=0;a<n;a++){
                                cout << "Gene " << a << ": " << genes_x[selections[2]*n+a] << " "<<genes_y[selections[2]*n+a] << endl;
                        }
			cout << endl << endl;
			*/
			crossover <<< blks, dim3(NUM_THREADS/chromo_size,chromo_size,1) >>> (d_genes_raw_x, d_genes_raw_y, d_offspring_x_raw, d_offspring_y_raw, d_selections_raw, chromo_size,population_size, crossover_points_raw);
			
			// Test -6 Printing the offspring to check the cross over
			/*thrust::copy(d_offspring_x.begin(),d_offspring_x.end(),offspring_x.begin());			
			thrust::copy(d_offspring_y.begin(),d_offspring_y.end(),offspring_y.begin());			
			for(int a=0;a<n;a++){
                                cout << "Offspring " << a << ": " << offspring_x[a] << " "<<offspring_y[a] << endl;
                        }			
			for(int a=2*n;a<3*n;a++){
                                cout << "Offspring " << a << ": " << offspring_x[a] << " "<<offspring_y[a] << endl;
                        }*/

			
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );

			//Creating 3*N*L random Numbers
			float* d_random2_raw = thrust::raw_pointer_cast(&d_random2[0]);
			thrust::transform(thrust::counting_iterator<int>(0),
					thrust::counting_iterator<int>(3*population_size*chromo_size),
					d_random2.begin(),
					RandomNumberFunctor(seed));
			//Mutation
			mutation<<<blks, dim3(NUM_THREADS/chromo_size,chromo_size,1) >>> (d_offspring_x_raw,d_offspring_y_raw,d_random2_raw,population_size,chromo_size);

			swap(d_genes_x,d_offspring_x);
			swap(d_genes_y,d_offspring_y);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );

		}
		//Find the best gene
		compute_fitness<<<blks,NUM_THREADS>>>(d_fitness_raw,d_genes_raw_x,d_genes_raw_y,observers_x,observers_y,targets_x,targets_y,observers_count,targets_count,radar_range);
		thrust::copy(d_fitness.begin(), d_fitness.end(), fitness.begin());
		thrust::copy(d_genes_x.begin(), d_genes_x.end(), genes_x.begin());
		thrust::copy(d_genes_y.begin(), d_genes_y.end(), genes_y.begin());
		//Observer position update
		int the_best = thrust::distance(fitness.begin(), thrust::max_element(fitness.begin(), fitness.begin() + n));
		if(fitness[the_best]>=0)
			for(int a=the_best*chromo_size;a<(the_best+1)*chromo_size;a++)
			{host_observers_x[a%n]+=genes_x[a];host_observers_y[a%n]+=genes_y[a];}
		// Target position update
		for(int a=0;a<m;a++){
			while(!(host_targets_x[a]+target_dpick_x[a]>=0 && host_targets_x[a]+target_dpick_x[a]<=150))
			target_dpick_x[a]=randint(-1,1);
			host_targets_x[a]+=target_dpick_x[a];
			while(!(host_targets_y[a]+target_dpick_y[a]>=0 && host_targets_y[a]+target_dpick_y[a]<=150))
				target_dpick_y[a]=randint(-1,1);
			host_targets_y[a]+=target_dpick_y[a];
		}
		//if(tim%50==0||((rand()%100)>=80)){
			for(int a=0;a<m;a++){target_dpick_x[a]=randint(-1,1);target_dpick_y[a]=randint(-1,1);}
		//}
		// Score update 
		int temp_score=0;
		for(int a=0;a<m;a++){
			int fl=0;
			for(int b=0;b<n;b++){
				if(sqrt((host_observers_x[b]-host_targets_x[a])*(host_observers_x[b]-host_targets_x[a])+(host_observers_y[b]-host_targets_y[a])*(host_observers_y[b]-host_targets_y[a]))<=radius)fl=1;
			}
			if(fl==1)temp_score+=1;
		}
		score=(score*tim+temp_score)/(tim+1);	
	}
	cudaFree(observers_x);
	cudaFree(observers_y);
	cudaFree(observers_count);
	cudaFree(targets_x);
	cudaFree(targets_y);	
	cudaFree(targets_count);
	cout << "observer_count = " << n << ", target_count = " << m << ", radius = " << radius << ", iteration = " << t << ", score = " << score << endl;
	//printf("Average Number of targets observed in each time step: %f\n",score);
}

