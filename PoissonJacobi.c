/* LAB COURSE II Project WS 18/19 - Jacobi Method for solving Poisson Equation in MPI.

Code for Jacobi Iteration.
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
#define maxIterations 50000
#define tolerance 0.00001

struct mylattice {
int globaln;
int totalprocs;
int totalrowprocs;
int totalcolprocs;
int myrank;
int myrowcoord;
int mycolcoord;
double gridlengthX;
double gridlengthY;
int localrows;
int localcols;
int localn;
double* leftbndry;
double* rightbndry;
double* upbndry;
double* downbndry;
double* phi;
double* phiold;
MPI_Status* status;
} me;


double find_x(struct mylattice *me, int localcolnumber, double h);
double find_y(struct mylattice *me, int localrownumber, double h);
double g(double x, double y){return (1+x) * sin(x+y);}

double f(struct mylattice *me, int localrownumber, int localcolnumber, double h){
	double x = find_x(me, localcolnumber, h);
	double y = find_y(me, localrownumber, h);
	return 2*( ((1+x)*sin(x+y)) - cos(x+y));}

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
void JacobiIteration(struct mylattice *me, double h);
void LocalSum(struct mylattice *me, double* localsum);
void ComputeTrueLocalDiff(struct mylattice *me, double h, double* truelocaldiff);
void ComputeLocalMax(struct mylattice *me, double* localmax);


int main(int argc, char* argv[]) {


int N = 36864;
int rowprocs = 4;
int colprocs = 4;







double initialguess = 0;
int iterations = 0;
double localmax;
double globalmax;
double localsum;
double globalsum;
double diffnorm;
double starttime;
double time;
double h = 1/(sqrt(N)+1);
double truelocaldiff;
double trueglobaldiff;

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
	JacobiIteration(&me,h);
	MPI_Barrier(comm_2D);
	ExchangeBoundaryPoints(&me, &comm_2D);
	MPI_Barrier(comm_2D);
	ComputeLocalMax(&me, &localmax);
	//LocalSum(&me, &localsum);
	copyPhi(&me);
	MPI_Allreduce(&localmax, &globalmax, 1, MPI_DOUBLE, MPI_SUM, comm_2D);
	//MPI_Allreduce(&localsum, &globalsum, 1, MPI_DOUBLE, MPI_SUM, comm_2D);
	//diffnorm = sqrt(globalsum)/N;
	}
	while(iterations < maxIterations && tolerance < globalmax);
	//while(iterations < maxIterations && tolerance < diffnorm);
	
	
MPI_Barrier(comm_2D); 
time = MPI_Wtime() - starttime;

ComputeTrueLocalDiff(&me, h, &truelocaldiff);

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
printf("Total Iterations(Jacobi) = %d\nTotal Grid Points = %d\nTotal Time(Jacobi) = %f\nTotal Processes: %d\n", iterations, N, time, me.totalprocs);
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


double find_x(struct mylattice *me, int localcolnumber, double h){  
  double x;
  return x = (me->mycolcoord)*(me->localcols)*h + localcolnumber*h;
}

double find_y(struct mylattice *me, int localrownumber, double h){
  double y;
  return y = (me->myrowcoord)*(me->localrows)*h + localrownumber*h;
}

int findrank(struct mylattice *me, int rowcoord, int colcoord){return (rowcoord*(me->totalcolprocs) + colcoord);}

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
		int j = 0;
		double val = g(find_x(me,j,h),find_y(me,i,h));
		me->phi[i*(me->localcols+2) + j] = val;
		me->phiold[i*(me->localcols+2) + j] = val;
		me->leftbndry[i-1] = val;
}
}

if (me->mycolcoord+1>me->totalcolprocs-1){
	for (i = 1; i<=me->localrows; i++){
		int j = me->localcols+1;
		double val = g(find_x(me,j,h),find_y(me,i,h));
		me->phi[i*(me->localcols+2) + j] = val;
		me->phiold[i*(me->localcols+2) + j] = val;
		me->rightbndry[i-1] = val;
}
}
if (me->myrowcoord-1 < 0){
	for (j = 1; j<=me->localcols; j++){
	int i = 0;
	double val = g(find_x(me,j,h),find_y(me,i,h));
	me->phi[i*(me->localcols+2) + j] = val;
	me->phiold[i*(me->localcols+2) + j] = val;
	me->upbndry[j-1] = val;
}
}
if (me->myrowcoord+1 > me->totalrowprocs-1){
	for (j = 1; j<=me->localcols; j++){
	int i = me->localrows+1;
	double val = g(find_x(me,j,h),find_y(me,i,h));
	me->phi[i*(me->localcols+2) + j] = val;
	me->phiold[i*(me->localcols+2) + j] = val;
	me->downbndry[j-1] = val;
}
}


}


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

void copyPhi(struct mylattice *me){
int i, j;
for (i = 0; i<=me->localrows+1; i++){
	for (j = 0; j<=me->localcols+1; j++){
		me->phiold[i*(me->localcols+2) + j] = me->phi[i*(me->localcols+2) + j];
		
	}
}
}


void JacobiIteration(struct mylattice *me, double h){
int i, j;
for (i = 1; i<=me->localrows; i++){
	for(j = 1; j<=me->localcols; j++){
		me->phi[i*(me->localcols+2) + j] = 0.25*(me->phiold[(i+1)*(me->localcols+2) + j] + me->phiold[(i-1)*(me->localcols+2) + j] +  			me->phiold[(i)*(me->localcols+2) + j+1] + me->phiold[(i)*(me->localcols+2) + j-1] + h*h*f(me,i,j,h));
}
}
}

void LocalSum(struct mylattice *me, double* localsum){
	*localsum = 0;
	int i, j;
	for (i = 1; i<=me->localrows; i++){
		for(j = 1; j<=me->localcols; j++){
			*localsum = *localsum + ( (me->phi[i*(me->localcols+2) + j]) - (me->phiold[i*(me->localcols+2) + j]) ) * ( (me->phi[i*(me->localcols+2) + j]) - (me->phiold[i*(me->localcols+2) + j]) );
		}
	}
}

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



void ComputeTrueLocalDiff(struct mylattice *me, double h , double* truelocaldiff){
	*truelocaldiff = 0;
	int i, j;
	for (i = 1; i<=me->localrows; i++){
		for(j = 1; j<=me->localcols; j++){
			*truelocaldiff = *truelocaldiff + ( (u(me,i,j,h) - (me->phi[i*(me->localcols+2) + j])) * (u(me,i,j,h) - (me->phi[i*(me->localcols+2) + j]))  );
		}
	}
}

void PrintValues(struct mylattice *me){
int i, j;
for (i = 1; i<=me->localrows; i++){
	for (j = 1; j<=me->localcols; j++){
		printf("%f ", me->phi[i*(me->localcols+2) + j]);		
		
	}
	printf("\n");
}

}






