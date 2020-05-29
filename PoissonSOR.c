/* LAB COURSE II Project WS 18/19 - SOR Method for solving Poisson Equation in MPI.

Code for Gauss Seidel with Successive Over-Relaxation.
Submitted by:

Ali Raza Ghafoor - 1667580
Muhammad Bilal - 1736928

*/






#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define maxIterations 40000
#define tolerance 0.00001
#define w 1.8 // Relaxation parameter


// Struct which contains all the information for each processor
struct mylattice {
int globaln; 		// Total number of grid points excluding boundaries
int totalprocs; 	// Total number of processors
int totalrowprocs;	// Total processors in row
int totalcolprocs;	// Total processors in column
int myrank;			// Rank in the new 2D communicator
int myrowcoord;		// Row coordinate of the processor
int mycolcoord;		// Column coordinate of the processor
double gridlengthX; // The total grid length in X direction that each processor is responsible for. Not used in the program.
double gridlengthY; // The total grid length in Y direction that each processor is responsible for. Not used in the program.
int localrows;		// Local rows that each processor has. Includes the local boundary rows.
int localcols;		// Local columns that each processor has. Includes the local boundary columns.
int localn;			// Total points that each processor has. Includes the points in the boundary ring.
double* leftbndry;	// Vector for storing left boundary.
double* rightbndry; // Vector for storing right boundary.
double* upbndry;	// Vector for storing upper boundary.
double* downbndry; 	// Vector for storing lower boundary.
double* phi;		// Vector which stores the values at local grid points (includes boundary values as well).
double* phiold;		// Vector for storing old values at local grid points during iteration.
MPI_Status* status;
} me;


double find_x(struct mylattice *me, int localcolnumber, double h);
double find_y(struct mylattice *me, int localrownumber, double h);
double g(double x, double y){return (1+x) * sin(x+y);} // Function for computing values at boundary points.

// Source function
double f(struct mylattice *me, int localrownumber, int localcolnumber, double h){
	double x = find_x(me, localcolnumber, h);
	double y = find_y(me, localrownumber, h);
	return 2*( ((1+x)*sin(x+y)) - cos(x+y));}

// Original solution function for comparison of norms after the iterative solution has been found at the end.
double u(struct mylattice *me, int localrownumber, int localcolnumber, double h){
	double x = find_x(me, localcolnumber, h);
	double y = find_y(me, localrownumber, h);
	return  ((1+x)*sin(x+y));}



void setlattinfo(struct mylattice *me, int N, int rowprocs, int colprocs);
void setgrid (struct mylattice *me, double h, double initialguess);
int findrank(struct mylattice *me, int rowcoord, int colcoord);
void sendbndry(struct mylattice *me, int flag, MPI_Comm* comm_2D);
void recvbndry(struct mylattice *me, int flag, MPI_Comm* comm_2D);
void ExchangeBoundaryPoints(struct mylattice *me, MPI_Comm* comm_2D);
void copyPhi(struct mylattice *me);
void PrintValues(struct mylattice *me);
void UpdateBoundaryValues(struct mylattice *me);
void SOR_RED(struct mylattice *me, double h);
void SOR_BLACK(struct mylattice *me, double h);
void LocalSum(struct mylattice *me, double* localsum);
void ComputeTrueLocalDiff(struct mylattice *me, double h, double* truelocaldiff);
void ComputeLocalMax(struct mylattice *me, double* localmax);


int main(int argc, char* argv[]) {

/* The following three paramters have to be set by the user. Conditions:
1) N should be divisible by total number of processors to be used.
2) Square root of N should be divisible by the number of processors in both the Rows and Columns */

int N = 36864;  // (192*192)	// Total grid points excluding the boundary points. Have to be set by the user.
int rowprocs = 4;	// Total processors in the Rows. Have to be set by the user.
int colprocs = 4;	// Total processors in the Columns. Have to be set by the user.







double initialguess = 0;
int iterations = 0;
double localmax;		// Stores the maximum difference at a grid point between old solution and new solution for individual processor.
double globalmax;		// Stores the maximum global difference for all processors. Used for stopping criteria.
double localsum;		// Not used anymore, but can be used to have a different stopping criteria (2 norm instead of max norm). 
double globalsum;		// Not used anymore, but can be used to have a different stopping criteria (2 norm instead of max norm).
double diffnorm;		// Not used anymore, but can be used to have a different stopping criteria (2 norm instead of max norm).
double starttime;		// For storing the starting time value.
double time;			// For storing the total time taken.
double h = 1/(sqrt(N)+1);	// Step size.
double truelocaldiff;	// Stores the sum of the squares of difference between true solution and iterative solution.
double trueglobaldiff;  // Used for calculating global 2 norm of the difference between iterative solution and true solution. 

MPI_Comm comm_2D;
int wrap_around[2];
int coords[2];
int dims[2];
dims[0] = rowprocs; dims[1] = colprocs;        
wrap_around[0] = wrap_around[1] = 0;
MPI_Init(&argc, &argv);

MPI_Cart_create(MPI_COMM_WORLD, 2, dims, wrap_around, 1, &comm_2D);
MPI_Comm_rank(comm_2D, &me.myrank);
MPI_Cart_coords(comm_2D, me.myrank, 2, coords);
me.myrowcoord = coords[0];  me.mycolcoord = coords[1];
MPI_Comm_size(comm_2D, &me.totalprocs); 


// Check if the conditions are fulfilled.
if( (sqrt(N)/rowprocs) != (int)(sqrt(N)/rowprocs) || (sqrt(N)/colprocs) != (int)(sqrt(N)/colprocs) || sqrt(N) != (int)sqrt(N) ){
	if (me.myrank == 0){
	printf("The square root of N must be an integer, and it must be divisible by the number of row processors and column processors.\n");
	}
MPI_Comm_free(&comm_2D);
MPI_Finalize();
return 0;
}
else{



setlattinfo(&me , N, rowprocs, colprocs);
setgrid(&me, h, initialguess);

MPI_Barrier(comm_2D);
starttime = MPI_Wtime();

do{
	iterations++;
	MPI_Barrier(comm_2D);
	
	SOR_RED(&me,h);  // Even points
	MPI_Barrier(comm_2D);
	ExchangeBoundaryPoints(&me, &comm_2D);
	MPI_Barrier(comm_2D);
	SOR_BLACK(&me,h); // Odd Points
	MPI_Barrier(comm_2D);
	ExchangeBoundaryPoints(&me, &comm_2D);
	ComputeLocalMax(&me, &localmax);
	//LocalSum(&me, &localsum); // can be uncommented if using a different stopping criteria. 
	copyPhi(&me);
	MPI_Allreduce(&localmax, &globalmax, 1, MPI_DOUBLE, MPI_SUM, comm_2D);
	//MPI_Allreduce(&localsum, &globalsum, 1, MPI_DOUBLE, MPI_SUM, comm_2D); // can be uncommented if using a different stopping criteria.
	//diffnorm = sqrt(globalsum)/N; // can be uncommented if using a difference stopping criteria.
	}
	while(iterations < maxIterations && tolerance < globalmax); // Stopping criteria using global max difference, as suggested in class.
	//while(iterations < maxIterations && tolerance < diffnorm); // Stopping criteria using 2-norm, not used anymore. 
	
	
MPI_Barrier(comm_2D); 
time = MPI_Wtime() - starttime;

ComputeTrueLocalDiff(&me, h, &truelocaldiff); // Computes 2 norm of difference between true solution and iterative solution.

MPI_Reduce(&truelocaldiff, &trueglobaldiff, 1, MPI_DOUBLE, MPI_SUM, 0, comm_2D);
	
if (me.myrank == 0){	

trueglobaldiff = sqrt(trueglobaldiff)/N;

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// This part prints out the X and Y values at each of the grid points and the Iterative solution at those points for all processors.  



	
	for (int i = 1; i<=me.localrows; i++){
		for(int j = 1; j<=me.localcols; j++){
			printf("x = %f, y = %f, solution = %f\n", find_x(&me,j,h), find_y(&me,i,h), me.phi[i*(me.localcols+2) + j]);
		}
	}
int tagX = 0, tagY = 1, tagSol = 2, count = 0;
int myN = me.localrows*me.localcols;
double* xValues = (double *)malloc((myN)*sizeof(double));
double* yValues = (double *)malloc((myN)*sizeof(double));
double* solValues = (double *)malloc((myN)*sizeof(double));



	for (int source = 1; source < me.totalprocs; source++){
		
		
		MPI_Recv(xValues, myN, MPI_DOUBLE, source, tagX, comm_2D, me.status);
		MPI_Recv(yValues, myN, MPI_DOUBLE, source, tagY, comm_2D, me.status);
		MPI_Recv(solValues, myN, MPI_DOUBLE, source, tagSol, comm_2D, me.status);
		
		printf("Solution values on grid points for process %d of total %d processes:\n", source, me.totalprocs);
		for (int k = 0; k<myN; k++){
			printf("x = %f, y = %f, solution = %f\n", xValues[k], yValues[k], solValues[k]);
		}
		
	}
free(xValues);
free(yValues);
free(solValues);
	

}



if (me.myrank != 0){
	int tagX = 0, tagY = 1, tagSol = 2, count = 0;
	int myN = me.localrows*me.localcols;
	double* xValues = (double *)malloc((myN)*sizeof(double));
	double* yValues = (double *)malloc((myN)*sizeof(double));
	double* solValues = (double *)malloc((myN)*sizeof(double));
	for (int i = 1; i<=me.localrows; i++){
		for(int j = 1; j<=me.localcols; j++){
			xValues[count] = find_x(&me,j,h);
			yValues[count] = find_y(&me,i,h); 
			solValues[count] = me.phi[i*(me.localcols+2) + j];
			count++;
		}
	} 
	
	MPI_Send(xValues, myN, MPI_DOUBLE, 0, tagX, comm_2D);		
	MPI_Send(yValues, myN, MPI_DOUBLE, 0, tagY, comm_2D);
	MPI_Send(solValues, myN, MPI_DOUBLE, 0, tagSol, comm_2D);
	free(xValues);
	free(yValues);
	free(solValues);
}


	
	

------------------------------------------------------------------------------------------------------------------------------------------------------------*/
printf("Total Iterations(GSOR) = %d\nTotal Grid Points = %d\nTotal Time(GSOR) = %f\nTotal Processes: %d\n", iterations, N, time, me.totalprocs);
printf("The 2-Norm of the difference between True Solution and Iterative Solution is: %f\n", trueglobaldiff);
}
free(me.phi);
free(me.phiold);
free(me.leftbndry);
free(me.rightbndry);
free(me.upbndry);
free(me.downbndry);
MPI_Comm_free(&comm_2D);
MPI_Finalize();
return 0;
}
}




// Function to store the local information for each processor.
void setlattinfo(struct mylattice *me, int N, int rowprocs, int colprocs){

me->globaln = N;
me->totalrowprocs = rowprocs;
me->totalcolprocs = colprocs;
me->gridlengthX = 1;
me->gridlengthY = 1;
me->localrows = sqrt(N)/rowprocs;
me->localcols = sqrt(N)/colprocs;
me->localn = (me->localrows+2) * (me->localcols+2);
me->leftbndry = (double *)malloc((me->localrows)*sizeof(double));
me->rightbndry = (double *)malloc((me->localrows)*sizeof(double));
me->upbndry = (double *)malloc((me->localcols)*sizeof(double));
me->downbndry = (double *)malloc((me->localcols)*sizeof(double));
me->phi = (double *)malloc(me->localn*sizeof(double));
me->phiold = (double *)malloc(me->localn*sizeof(double));

}

// Function to find X values at the local grid points for each processor
double find_x(struct mylattice *me, int localcolnumber, double h){  
  double x;
  return x = (me->mycolcoord)*(me->localcols)*h + localcolnumber*h;
}
// Function to find Y values at the local grid points for each processor
double find_y(struct mylattice *me, int localrownumber, double h){
  double y;
  return y = (me->myrowcoord)*(me->localrows)*h + localrownumber*h;
}

// Function to find the 2D grid rank of each processor
int findrank(struct mylattice *me, int rowcoord, int colcoord){return (rowcoord*(me->totalcolprocs) + colcoord);}


// Function that initializes the solution vectors to initial guess, and sets up the global boundary values at the appropriate processors.
void setgrid (struct mylattice *me, double h, double initialguess){
int i, j;
for (i = 0; i<=me->localrows+1; i++){
	for (j = 0; j<=me->localcols+1; j++){
		me->phi[i*(me->localcols+2) + j] = initialguess;
		me->phiold[i*(me->localcols+2) + j] = initialguess;
		
	}
}


if (me->mycolcoord-1<0){
	for (i = 1; i<=me->localrows; i++){
		j = 0;
		double val = g(find_x(me,j,h),find_y(me,i,h));
		me->phi[i*(me->localcols+2) + j] = val;
		me->phiold[i*(me->localcols+2) + j] = val;
		me->leftbndry[i-1] = val;
}
}

if (me->mycolcoord+1>me->totalcolprocs-1){
	for (i = 1; i<=me->localrows; i++){
		j = me->localcols+1;
		double val = g(find_x(me,j,h),find_y(me,i,h));
		me->phi[i*(me->localcols+2) + j] = val;
		me->phiold[i*(me->localcols+2) + j] = val;
		me->rightbndry[i-1] = val;
}
}
if (me->myrowcoord-1 < 0){
	for (j = 1; j<=me->localcols; j++){
	i = 0;
	double val = g(find_x(me,j,h),find_y(me,i,h));
	me->phi[i*(me->localcols+2) + j] = val;
	me->phiold[i*(me->localcols+2) + j] = val;
	me->upbndry[j-1] = val;
}
}
if (me->myrowcoord+1 > me->totalrowprocs-1){
	for (j = 1; j<=me->localcols; j++){
	i = me->localrows+1;
	double val = g(find_x(me,j,h),find_y(me,i,h));
	me->phi[i*(me->localcols+2) + j] = val;
	me->phiold[i*(me->localcols+2) + j] = val;
	me->downbndry[j-1] = val;
}
}


}

// Function that sends updated boundary values to neighbouring processors.
void sendbndry(struct mylattice *me, int flag, MPI_Comm* comm_2D){
int i, j, k;
if (flag == 1){ //Lower Boundary

double* sendtemp = (double *)malloc(me->localcols*sizeof(double));
i = me->localrows;
k = 0;
for (j = 1; j<= me->localcols; j++){
	sendtemp[k] = me->phi[i*(me->localcols+2) + j];
	k = k+1;
}

MPI_Send(sendtemp, me->localcols, MPI_DOUBLE, findrank(me, me->myrowcoord+1, me->mycolcoord) , 0, *comm_2D);

free(sendtemp);
}

if (flag == 2){ //Upper Boundary

double* sendtemp = (double *)malloc(me->localcols*sizeof(double));
i = 1;
k = 0;
for (j = 1; j<= me->localcols; j++){
	sendtemp[k] = me->phi[i*(me->localcols+2) + j];
	k=k+1;
}

MPI_Send(sendtemp, me->localcols, MPI_DOUBLE, findrank(me, me->myrowcoord-1, me->mycolcoord) , 0, *comm_2D);

free(sendtemp);
}


if (flag == 3){ //Left Boundary

double* sendtemp = (double *)malloc(me->localrows*sizeof(double));
j = 1;
k = 0;
for (i = 1; i<= me->localrows; i++){
	sendtemp[k] = me->phi[i*(me->localcols+2) + j];
	k=k+1;
}

MPI_Send(sendtemp, me->localrows, MPI_DOUBLE, findrank(me, me->myrowcoord, me->mycolcoord-1) , 0, *comm_2D);

free(sendtemp);
}

if (flag == 4){ //Right Boundary

double* sendtemp = (double *)malloc(me->localrows*sizeof(double));
j = me->localcols;
k = 0;
for (i = 1; i<= me->localrows; i++){
	sendtemp[k] = me->phi[i*(me->localcols+2) + j];
	k=k+1;
}

MPI_Send(sendtemp, me->localrows, MPI_DOUBLE, findrank(me, me->myrowcoord, me->mycolcoord+1) , 0, *comm_2D);

free(sendtemp);
}
}

// Function to receive boundaries from neighbouring processors
void recvbndry(struct mylattice *me, int flag, MPI_Comm* comm_2D){
if (flag == 1){ //Lower Boundary
MPI_Recv(me->downbndry, me->localcols, MPI_DOUBLE, findrank(me, me->myrowcoord+1, me->mycolcoord) , 0, *comm_2D, me->status);
}

if (flag == 2){ //Upper Boundary
MPI_Recv(me->upbndry, me->localcols, MPI_DOUBLE, findrank(me, me->myrowcoord-1, me->mycolcoord) , 0, *comm_2D, me->status);
}

if (flag == 3){ //Left Boundary
MPI_Recv(me->leftbndry, me->localrows, MPI_DOUBLE, findrank(me, me->myrowcoord, me->mycolcoord-1) , 0, *comm_2D, me->status);
}

if (flag == 4){ //Right Boundary
MPI_Recv(me->rightbndry, me->localrows, MPI_DOUBLE, findrank(me, me->myrowcoord, me->mycolcoord+1) , 0, *comm_2D, me->status);
}
}

// Function to update the boundary values once they've been received.
void UpdateBoundaryValues(struct mylattice *me){
// Lower Boundary
int i = me->localrows+1;
int k = 0;
int j;
for (j = 1; j<= me->localcols; j++){
	me->phi[i*(me->localcols+2) + j] = me->downbndry[k];
	k++;
}
// Upper Boundary
i = 0;
k = 0;
for (j = 1; j<= me->localcols; j++){
	me->phi[i*(me->localcols+2) + j] = me->upbndry[k];
	k++;
}
// Left Boundary
j = 0;
k = 0;
for (i = 1; i<= me->localrows; i++){
	me->phi[i*(me->localcols+2) + j] = me->leftbndry[k];
	k++;
}
// Right Boundary
j = me->localcols+1;
k = 0;
for (i = 1; i<= me->localrows; i++){
	me->phi[i*(me->localcols+2) + j] = me->rightbndry[k];
	k++;
}
}


/*
Function that exchanges the boundary values and updates them. It uses the functions sendbndry, recvbndry, UpdateBoundaryValues.
The boundary points are exchanged by numbering the processors into odd and even, to avoid any potential deadlock. Even processors
send the boundary vector first and the neighbouring odd processor receives it. Then the odd processor sends the boundary vector and so on.
*/

void ExchangeBoundaryPoints(struct mylattice *me, MPI_Comm* comm_2D){

int lowerbndry = 1;
int upperbndry = 2;
int leftbndry = 3;
int rightbndry = 4;


// for sending left boundary
if (me->mycolcoord -1 >0 && ((me->mycolcoord%2)==0)){
sendbndry(me, leftbndry , comm_2D);
}
if (me->mycolcoord +1 < me->totalcolprocs && ((me->mycolcoord%2)!=0)){
recvbndry(me, rightbndry , comm_2D);
}
if ((me->mycolcoord%2)!=0){
sendbndry(me, leftbndry , comm_2D);
}
if (me->mycolcoord+1<me->totalcolprocs && ((me->mycolcoord%2)==0)){
recvbndry(me, rightbndry , comm_2D);
}

// for sending right boundary
if (me->mycolcoord +1 < me->totalcolprocs  && ((me->mycolcoord%2)==0)){
sendbndry(me, rightbndry , comm_2D);
}
if ((me->mycolcoord%2)!=0){
recvbndry(me, leftbndry , comm_2D);
}
if (me->mycolcoord +1 < me->totalcolprocs && ((me->mycolcoord%2)!=0)){
sendbndry(me, rightbndry , comm_2D);
}
if (me->mycolcoord -1 > 0 && ((me->mycolcoord%2)==0)){
recvbndry(me, leftbndry , comm_2D);
}
// for sending lower boundary
if (me->myrowcoord +1 < me->totalrowprocs  && ((me->myrowcoord%2)==0)){
sendbndry(me, lowerbndry , comm_2D);
}
if ((me->myrowcoord%2)!=0){
recvbndry(me, upperbndry , comm_2D);
}
if (me->myrowcoord +1 < me->totalrowprocs && ((me->myrowcoord%2)!=0)){
sendbndry(me, lowerbndry , comm_2D);
}
if (me->myrowcoord -1 > 0 && ((me->myrowcoord%2)==0)){
recvbndry(me, upperbndry , comm_2D);
}
// for sending upper boundary
if (me->myrowcoord -1 >0 && ((me->myrowcoord%2)==0)){
sendbndry(me, upperbndry , comm_2D);
}
if (me->myrowcoord +1 < me->totalrowprocs && ((me->myrowcoord%2)!=0)){
recvbndry(me, lowerbndry , comm_2D);
}
if ((me->myrowcoord%2)!=0){
sendbndry(me, upperbndry , comm_2D);
}
if (me->myrowcoord+1<me->totalrowprocs && ((me->myrowcoord%2)==0)){
recvbndry(me, lowerbndry , comm_2D);
}

UpdateBoundaryValues(me);

}

// Function that copies the new solution into the old solution vector.
void copyPhi(struct mylattice *me){
int i, j;
for (i = 0; i<=me->localrows+1; i++){
	for (j = 0; j<=me->localcols+1; j++){
		me->phiold[i*(me->localcols+2) + j] = me->phi[i*(me->localcols+2) + j];
		
	}
}
}

// SOR iteration over Even points
void SOR_RED(struct mylattice *me, double h){
int i, j;
for (i = 1; i<=me->localrows; i++){
	for(j = 1; j<=me->localcols; j++){
		if((i + j) % 2 == 0){
		me->phi[i*(me->localcols+2) + j] = (1-w)*(me->phiold[i*(me->localcols+2) + j]) + w*0.25*(me->phi[(i+1)*(me->localcols+2) + 	j] + me->phi[(i-1)*(me->localcols+2) + j] +  
			me->phi[(i)*(me->localcols+2) + j+1] + me->phi[(i)*(me->localcols+2) + j-1] + h*h*f(me,i,j,h));
		}
	}
}
}

// SOR iteration over Odd points
void SOR_BLACK(struct mylattice *me, double h){
int i, j;
for (i = 1; i<=me->localrows; i++){
	for(j = 1; j<=me->localcols; j++){
		if((i + j) % 2 != 0){
		me->phi[i*(me->localcols+2) + j] = (1-w)*(me->phiold[(i)*(me->localcols+2) + j]) + w*0.25*(me->phi[(i+1)*(me->localcols+2) + 	j] + me->phi[(i-1)*(me->localcols+2) + j] +  
			me->phi[(i)*(me->localcols+2) + j+1] + me->phi[(i)*(me->localcols+2) + j-1] + h*h*f(me,i,j,h));
		}
	}
}
}

// Function that computes the square of difference between true solution and iterative solution over all the local grid points
void ComputeTrueLocalDiff(struct mylattice *me, double h , double* truelocaldiff){
	*truelocaldiff = 0;
	int i, j;
	for (i = 1; i<=me->localrows; i++){
		for(j = 1; j<=me->localcols; j++){
			*truelocaldiff = *truelocaldiff + ( (u(me,i,j,h) - (me->phi[i*(me->localcols+2) + j])) * (u(me,i,j,h) - (me->phi[i*(me->localcols+2) + j]))  );
		}
	}
}

// This function is not used anymore but can be used if using a different stopping criteria.
void LocalSum(struct mylattice *me, double* localsum){
	*localsum = 0;
	int i, j;
	for (i = 1; i<=me->localrows; i++){
		for(j = 1; j<=me->localcols; j++){
			*localsum = *localsum + ( (me->phi[i*(me->localcols+2) + j]) - (me->phiold[i*(me->localcols+2) + j]) ) * ( (me->phi[i*(me->localcols+2) + j]) - (me->phiold[i*(me->localcols+2) + j]) );
		}
	}
}

// Computes the local maximum between previous solution and new solution over all local grid points (excluding local boundaries).
void ComputeLocalMax(struct mylattice *me, double* localmax){
	int n = me->localrows*me->localcols;
	double* temparray = (double *)malloc(n*sizeof(double));
	int count = 0;
	int i,j,k;
	for (i = 1; i<=me->localrows; i++){
		for(j = 1; j<=me->localcols; j++){
			temparray[count] = fabs( (me->phi[i*(me->localcols+2) + j]) - (me->phiold[i*(me->localcols+2) + j]) );
			count++;
		}
	}
	double maximum = temparray[0];
	for (k = 0; k < n; k++){
		if (temparray[k] > maximum){
			maximum  = temparray[k];
		}
	}
	
	*localmax = maximum;
	free(temparray);
	
}

// Function the prints the values of the solution. Not used anymore, but can be used for printing solutions at all grid points of all the processors.
void PrintValues(struct mylattice *me){
int i, j;
for (i = 1; i<=me->localrows; i++){
	for (j = 1; j<=me->localcols; j++){
		printf("%f ", me->phi[i*(me->localcols+2) + j]);		
		
	}
	printf("\n");
}

}



