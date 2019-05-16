# GPU-Accelerated-Genetic-Algorithm-for-Co-Operative-Target-Observation
Problem Statement: Given a set of observers and targets, with each observer able to observe targets only in the feasible radius around it, objective is to maximise the average number of unique targets observed by all the observers. Genetic Algorithm is applied to this problem using a variety of genetic operators. Due to heavy execution time constraints, the algorithm has been modified to a more parallel version suitable to run on GPU .
</br></br>
<b>Commands for running </b></br>
<ul>
<li> For running as a batch file : sbatch batch.sh </li>
<li> For running individually 
  <ul> 
  <li> Compiling: nvcc honours-final.cu </li>
  <li> Getting the gpu: sinteractive -c 2 -g 1 </li>
  <li> Running it on gpu: ./a.out Arugements </li>
  </ul>
</ul>
<b>File descriptions:</b></br>
<ul>
<li>honours-final.cu : <p> The main code </p>  </li>
<li>output.txt : <p> Contains the output for the Parallel Genetic Algorithm with 150 iterations per genetic algorithm per time step with the data from Classic CTO.</p> </li>
<li>output-2.txt: <p> Contains the output for the Parallel Genetic Algorithm with 200 iterations per genetic algorithm per time step with the data from Classic CTO.</p> </li>
<li>output-serial.txt : <p> Contains the output for the Parallel Genetic Algorithm with 50 iterations per genetic algorithm per time step with the data from Classic CTO.</p> </li>
<li>output-serial2.txt : <p> Contains the output for the Parallel Genetic Algorithm with 25 iterations per genetic algorithm per time step with the data from Classic CTO.</p> </li>
<li>output-CTO-KMeans.txt : <p> Contains the output for the Parallel Genetic Algorithm with 300 iterations per genetic algorithm per time step with the data from KMeans Paper.</p> </li>
<li>batch-final.sh : <p> Conatins the commands for compiling and running the code on the ada gpu.</p> </li>
</ul>

