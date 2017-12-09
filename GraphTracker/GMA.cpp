#include "GMA.h"
#include <iostream>
#include <time.h>

GMA::GMA()
{
}


GMA::~GMA()
{
}

cv::Mat GMA::normalize(cv::Mat& x, cv::Mat& group)
{
	cv::Mat sum = x.t() * group * group.t();
	sum += DBL_MINVAL;

	return x / sum.t();
}

cv::Mat GMA::biNormalize(cv::Mat& x, vector<MatchInfo>& matches, int n1, int n2, int nIter)
{
	double* sum1 = new double[n1];
	double* sum2 = new double[n2];

	cv::Mat p = x.clone();

	for (int iter = 0; iter < nIter; iter++) {
		memset(sum1, 0, sizeof(double) * n1);
		for (int k = 0; k < matches.size(); k++) {
			sum1[matches[k].idx_g1] += p.at<double>(k, 0);
		}
		for (int k = 0; k < matches.size(); k++) {
			p.at<double>(k, 0) = p.at<double>(k, 0) / (sum1[matches[k].idx_g1] + DBL_MINVAL);
		}

		memset(sum2, 0, sizeof(double) * n2);
		for (int k = 0; k < matches.size(); k++) {
			sum2[matches[k].idx_g2] += p.at<double>(k, 0);
		}
		for (int k = 0; k < matches.size(); k++) {
			p.at<double>(k, 0) = p.at<double>(k, 0) / (sum2[matches[k].idx_g2] + DBL_MINVAL);
		}
	}

	delete[]sum1;
	delete[]sum2;

	return p;
}


cv::Mat GMA::greedyMapping(cv::Mat& x, vector<MatchInfo>& matches)
{
	double* x_ = new double[x.rows];
	for (int i = 0; i < x.rows; i++)
		x_[i] = x.at<double>(i, 0);

	cv::Mat y = cv::Mat_<double>::zeros(x.rows, 1);

	double maxVal = 0.0;
	int maxIdx = 0;
	for (int i = 0; i < x.rows; i++) {
		if (x_[i] > maxVal) {
			maxVal = x_[i];
			maxIdx = i;
		}
	}

	while (maxVal > 0.0) {
		y.at<double>(maxIdx, 0) = 1.0;

		int idx_g1 = matches[maxIdx].idx_g1;
		int idx_g2 = matches[maxIdx].idx_g2;
		for (int i = 0; i < matches.size(); i++) {
			if (matches[i].idx_g1 == idx_g1 || matches[i].idx_g2 == idx_g2){
				x_[i] = 0.0;
			}
		}

		maxVal = 0.0;
		maxIdx = 0;
		for (int i = 0; i < x.rows; i++) {
			if (x_[i] > maxVal) {
				maxVal = x_[i];
				maxIdx = i;
			}
		}
	}

	delete[]x_;

	return y;
}


cv::Mat GMA::PSM(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2)
{
	int max_iters = 100;
	double ro = 1e-6;

	int rows = affinity.rows;
	int n1 = group1.cols;
	int n2 = group2.cols;

	cv::Mat A = affinity;
	cv::Mat x = cv::Mat_<double>::ones(rows, 1);
	x = normalize(x, group1);

	int iter = 0;
	while (iter < max_iters) {
		cv::Mat x_ = x + DBL_MINVAL;

		x = A * x;
		if (iter % 2 == 0)
			x = normalize(x, group1);
		else
			x = normalize(x, group2);

		A = A.mul((x / x_) * cv::Mat_<double>::ones(1, rows));
		iter++;

		double err = 0;
		for (int row = 0; row < x.rows; row++) {
			double dif = x.at<double>(row, 0) - x_.at<double>(row, 0);
			err += dif * dif;
		}
		if (sqrt(err) / rows < ro)
			break;
	}

	x = greedyMapping(x, matches);


	return x;
}

cv::Mat GMA::SM(cv::Mat& affinity)
{

	cv::Mat x = cv::Mat_<double>::ones(affinity.rows, 1);
	x = x / cv::norm(x);

	int maxIter = 100;
	for (int i = 0; i < maxIter; i++) {
		cv::Mat_<double> x_ = x.clone();
		x = affinity * x;
		x = x / cv::norm(x);
		if (cv::norm(x - x_) < 1e-15)
			break;
	}

	return x;

}

cv::Mat GMA::IPFP(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2, cv::Mat& x0)
{
	// function parameter
	int nItMa = 50;

	cv::Mat  M = affinity;

	cv::Mat new_sol = x0.clone();
	cv::Mat best_sol = x0.clone();
	double best_score = 0;
	int nSteps = 0;

	while (nSteps <= nItMa) {
		nSteps = nSteps + 1;

		cv::Mat old_sol = new_sol.clone();
		cv::Mat xx = M * old_sol;

		// projection to discrete domain
		cv::Mat x2 = greedyMapping(xx, matches);

		// step size
		cv::Mat_<double> D = (x2 - old_sol).t() * M * (x2 - old_sol);
		if (D(0, 0) >= 0.0) {
			new_sol = x2;
		}
		else {
			cv::Mat_<double>  C = old_sol.t() * M * (x2 - old_sol);
			double r = min(1.0, -C(0, 0) / D(0, 0));
			if (r < 0.01) {
				r = 0;
			}
			else if (r == 1) {
				//	discreteRate = discreteRate + 1;
			}
			new_sol = old_sol + r * (x2 - old_sol);
		}

		cv::Mat_<double> curr_score = x2.t() * M * x2;

		if (curr_score(0, 0) > best_score) {
			best_score = curr_score(0, 0);
			best_sol = x2.clone();
		}

		// stop condition
		if (cv::norm(new_sol - old_sol) < 1e-15) {
			break;
		}
	}

	return best_sol;
}

cv::Mat GMA::IPFP(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2)
{
	// initial
	cv::Mat x0 = SM(affinity);

	return IPFP(affinity, matches, group1, group2, x0);
}

void GMA::bistocNorm(double* pX, int N,
	int* pIdx1, int* pID1, int* pIdx2, int* pID2,
	double* pTol, double* pDumDim, double* pDumVal, int* pMaxIter,
	double* pY)
{
	double tol = (*pTol);
	tol = tol*tol;
	double delta = tol;

	int i, nG1, nG2, iter;
	double *pX2 = new double[N];
	double *pTemp = new double[N];

	double tempSum = 0;

	// Num of Groups
	nG1 = pID1[N - 1];
	nG2 = pID2[N - 1];

	// MATLAB idx --> C++ idx
	//     for(i = 0; i < N; i++) pIdx1[i]--;
	//     for(i = 0; i < N; i++) pIdx2[i]--;

	iter = 0;

	//     printf("%f\t%f\n", delta, tol);
	//     printf("%d\t%d\n", iter, (*pMaxIter));
	//     

	while (delta >= tol && iter < (*pMaxIter))
	{
		iter++;

		// copy the current state
		for (i = 0; i < N; i++)
			pX2[i] = pX[i];

		// update domain 1
		pTemp[0] = pX[pIdx1[0]];
		for (i = 1; i < N; i++)
		{
			if (pID1[i] == pID1[i - 1])
				pTemp[i] = pTemp[i - 1] + pX[pIdx1[i]];
			else
				pTemp[i] = pX[pIdx1[i]];
		}
		for (i = N - 2; i >= 0; i--)
		{
			if (pID1[i] == pID1[i + 1])
				pTemp[i] = pTemp[i + 1];
		}

		for (i = 0; i < N; i++)
		{
			pX[pIdx1[i]] /= pTemp[i];
		}

		if ((*pDumDim) == 1)
		{
			for (i = N - nG2; i < N; i++)
				pX[i] *= (*pDumVal);
		}

		/*for(i = 0; i < N; i++)
		printf("%d %d %f\n", pID1[i], pIdx1[i]+1, pX[pIdx1[i]]);*/

		// update domain 2
		pTemp[0] = pX[pIdx2[0]];
		for (i = 1; i < N; i++)
		{
			if (pID2[i] == pID2[i - 1])
				pTemp[i] = pTemp[i - 1] + pX[pIdx2[i]];
			else
				pTemp[i] = pX[pIdx2[i]];
		}
		for (i = N - 2; i >= 0; i--)
		{
			if (pID2[i] == pID2[i + 1])
				pTemp[i] = pTemp[i + 1];
		}
		for (i = 0; i < N; i++)
		{
			pX[pIdx2[i]] /= pTemp[i];
		}

		if ((*pDumDim) == 2)
		{
			for (i = N - nG1; i < N; i++)
				pX[i] *= (*pDumVal);
		}

		/*printf("\n");
		for(i = 0; i < N; i++)
		printf("%d %d %f\n", pID2[i], pIdx2[i]+1, pX[pIdx2[i]]);
		printf("\n");*/

		// check the difference for termination criterion
		delta = 0;
		for (i = 0; i < N; i++)
		{
			delta += (pX[i] - pX2[i])*(pX[i] - pX2[i]);
		}
	}

	// return solution
	for (i = 0; i < N; i++)
		pY[i] = pX[i];

	//     printf("Iter: %d\n", iter);
	//     
	//     for(i = 0; i < N; i++)
	//         printf("%f\t%f\n", pX[i], pY[i]);

	delete[] pX2;
	delete[] pTemp;
}


void GMA::make_groups_slack(const vector<MatchInfo>& matches, const cv::Mat& group1, const cv::Mat& group2,
	double& dumDim, double& dumVal, int& dumSize, int* idx1, int* ID1, int* idx2, int* ID2)
{
	int n1 = group1.cols;
	int n2 = group2.cols;
	int nn = matches.size();

	vector<MatchInfo> exMatches;
	exMatches.assign(matches.begin(), matches.end());

	if (n1 < n2) {
		dumDim = 1;
		dumVal = n2 - n1;
		dumSize = n2;

		for (int i = 0; i < n2; i++) {
			MatchInfo info;
			info.idx_match = nn + i;
			info.idx_g1 = n1;
			info.idx_g2 = i;
			info.score = dumVal;
			exMatches.push_back(info);
		}
	}
	else if (n1 > n2) {
		dumDim = 2;
		dumVal = n1 - n2;
		dumSize = n1;

		for (int i = 0; i < n1; i++) {
			MatchInfo info;
			info.idx_match = nn + i;
			info.idx_g1 = i;
			info.idx_g2 = n2;
			info.score = dumVal;
			exMatches.push_back(info);
		}
	}
	else {
		dumDim = 0;
		dumVal = 0;
		dumSize = 0;
	}

	int m_index = 0;
	int n_index = 1;
	for (int node = 0; node <= n1; node++) {
		int n_num = 0;
		for (int k = 0; k < exMatches.size(); k++) {
			if (exMatches[k].idx_g1 == node) {
				idx1[m_index] = k;
				ID1[m_index] = n_index;
				m_index++;
				n_num++;
			}
		}
		if (n_num > 0) {
			n_index++;
		}
	}

	m_index = 0;
	n_index = 1;
	for (int node = 0; node <= n2; node++) {
		int n_num = 0;
		for (int k = 0; k < exMatches.size(); k++) {
			if (exMatches[k].idx_g2 == node) {
				idx2[m_index] = k;
				ID2[m_index] = n_index;
				m_index++;
				n_num++;
			}
		}
		if (n_num > 0) {
			n_index++;
		}
	}



	return;
}


cv::Mat GMA::RRWM(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2)
{
	// get groups for bistochastic normalization
	//	[idx1 ID1] = make_groups(group1);
	//[idx2 ID2] = make_groups(group2);

	clock_t t[4] = { 0 };


	int n1 = group1.cols;
	int n2 = group2.cols;
	int nn = group1.rows;
	int nAdd = (n1 == n2) ? 0 : max(n1, n2);

	int* idx1 = new int[nn + nAdd];
	int* ID1 = new int[nn + nAdd];
	int* idx2 = new int[nn + nAdd];
	int* ID2 = new int[nn + nAdd];
	double *pX = new double[nn + nAdd];
	double *pY = new double[nn + nAdd];

	clock_t t0 = clock();

	double dumDim = 0, dumVal = 0;
	int dumSize = 0;
	make_groups_slack(matches, group1, group2, dumDim, dumVal, dumSize, idx1, ID1, idx2, ID2);

	t[0] = clock() - t0;


	// note that this matrix is column - wise stochastic
	double maxD = 0.0;
	for (int i = 0; i < affinity.cols; i++) {
		double d = cv::sum(affinity.col(i)).val[0];
		if (d > maxD)
			maxD = d;
	}
	cv::Mat Mo = affinity / maxD;

	// initialize answer
	int nMatch = affinity.rows;
	cv::Mat prev_score = cv::Mat_<double>::ones(affinity.rows, 1) / nMatch;		// buffer for the previous score
	cv::Mat	prev_score2 = prev_score.clone();         // buffer for the two iteration ahead
	cv::Mat	prev_assign = prev_score.clone();		// buffer for Sinkhorn result of prev_score
	cv::Mat cur_score = prev_score.clone();

	int bCont = 1;  
	int iter_i = 0;

	// for convergence check of power iteration
	double thresConvergence = 1e-25;
	double 	thresConvergence2 = nMatch * cv::norm(affinity, CV_L1) * DBL_MINVAL;
	cv::Mat LA = prev_score.t() * affinity * prev_score;
	double la = LA.at<double>(0, 0);

	double c = 0.2;
	double amp_max = 30;
	int iterMax = 300;
	while (bCont && iter_i < iterMax) {
		iter_i = iter_i + 1;

		t0 = clock();

		// random walking with reweighted jumps
		cur_score = Mo * ( c * prev_score + (1 - c) * prev_assign);

		t[1] += clock() - t0;

		t0 = clock();

		double sumCurScore = cv::sum(cur_score).val[0];		// normalization of sum 1
		if (sumCurScore > 0) {
			cur_score = cur_score / sumCurScore;
		}

		// update reweighted jumps
		cv::Mat cur_assign = cur_score.clone();

		// attenuate small values and amplify large values
		double cur_max = 0.0;
		for (int row = 0; row < cur_assign.rows; row++) {
			if (cur_assign.at<double>(row, 0) > cur_max)
				cur_max = cur_assign.at<double>(row, 0);
		}
		double amp_value = amp_max / cur_max;		// compute amplification factor
		cv::Mat cvTmp = amp_value*cur_assign;
		cv::exp(cvTmp, cur_assign);
	//	cur_assign = cv::exp(amp_value*cur_assign);

		t[2] += clock() - t0;

		t0 = clock();

		// Sinkhorn method of iterative bistocastic normalizations
	//	X_slack = [cur_assign; dumVal*ones(dumSize, 1)];
	//	X_slack = mexBistocNormalize_match_slack(X_slack, int32(idx1), int32(ID1), int32(idx2), int32(ID2), tolC, dumDim, dumVal, int32(1000));
	//	cur_assign = X_slack(1:nMatch);


		for (int i = 0; i < nn; i++)
			pX[i] = cur_assign.at<double>(i, 0);
		for (int i = 0; i < nAdd; i++)
			pX[nn + i] = dumVal;

		int N = nn + nAdd;
		double tolC = 1e-3;
		int nMaxIter = 100;
		bistocNorm(pX, N, idx1, ID1, idx2, ID2, &tolC, &dumDim, &dumVal, &nMaxIter, pY);

		for (int i = 0; i < nn; i++)
			cur_assign.at<double>(i, 0) = pY[i];


		t[3] += clock() - t0;

	//	cur_assign = biNormalize(cur_assign, matches, group1.cols, group2.cols, 10);

		double sumCurAssign = cv::sum(cur_assign).val[0];		// normalization of sum 1
		if (sumCurAssign > 0.0) {
			cur_assign = cur_assign / sumCurAssign;
		}

		// Check the convergence of random walks
		//		diff1 = sum((cur_score - prev_score). ^ 2);
		//		diff2 = sum((cur_score - prev_score2). ^ 2);	// to prevent oscillations
		double diff1 = (cur_score - prev_score).dot(cur_score - prev_score);
		double diff2 = (cur_score - prev_score2).dot(cur_score - prev_score2);	// to prevent oscillations
		double diff_min = min(diff1, diff2);
		if (diff_min < thresConvergence) {
			bCont = 0;
		}

		prev_score2 = prev_score.clone();
		prev_score = cur_score.clone();
		prev_assign = cur_assign.clone();

	} // end of main iteration

	delete[]idx1;
	delete[] ID1;
	delete[] idx2;
	delete[] ID2;
	delete[] pX;
	delete[] pY;

//	printf("\r\nRRWM: %d, %d, %d, %d", t[0], t[1], t[2], t[3]);

	return cur_score;
}

cv::Mat GMA::GNCCP_APE(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2)
{
	int nGroups1 = group1.cols;
	int nGroups2 = group2.cols;
	int nMatches = matches.size();

	// parameters
	double deta = 0.01;		// for the step length
	double eta = 0.001;     // for FW algorithm thresold
	int nHist = 10;			// number of recorded history

	// initialize 
	cv::Mat K = -affinity;
	cv::Mat x = cv::Mat_<double>::ones(nMatches, 1);
	x = normalize(x, group1);

	vector<cv::Mat> x_store(nHist, cv::Mat());
	vector<int> sLen_store(nHist, 1);
	cv::Mat x_est;

	double gamma = 1.0;
	int sLen = 1;
	int count = 0;
	while (gamma > -1.0) {
		// the Frank_Wolfe algorithm
		x = FW_GNCCP(x, K, matches, group1, group2, gamma);

		// check for convergency of gamma
		double nor = cv::norm(x);
		if ((cv::max(nGroups1, nGroups2) - nor * nor) < eta)
			break;

		// adjust step length
		if (count >= nHist)
			sLen = AdaptiveStep(x, x_est, sLen);

		gamma = gamma - sLen * deta;

		// store history solutions
		x_store.erase(x_store.begin());
		x_store.push_back(x);

		// store history step length
		sLen_store.erase(sLen_store.begin());
		sLen_store.push_back(sLen);

		// estimate next solution x as the start point of next iteraton
		count = count + 1;
		if (count >= nHist) {
			x_est = EstimateByTangent(x_store, sLen_store, 1);
			x_est = normalize(x_est, group1);

			if (Obj_GNCCP(x_est, K, gamma) < Obj_GNCCP(x, K, gamma))
				x = x_est;
		}
	}

	return x;
}

int GMA::AdaptiveStep(cv::Mat& x, cv::Mat& x_est, int sLen)
{
	double theta = 0.01;	// for the adaptive path estimation thresold
	int rho = 2;			// for the step length growth rate
	cv::Mat_<double> tr = (x_est - x).t() * (x_est - x);
	if (tr(0, 0) < theta)
		sLen = rho * sLen;
	else
		sLen = max(1, sLen / rho);
	return sLen;
}

cv::Mat GMA::FW_GNCCP(cv::Mat& x, cv::Mat& K, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2, double gamma)
{
	// parameters
	double eta = 0.001;     // for FW algorithm thresold
	int nItMa = 100;		// maximum iterations for FW algorithm

	int nGroup1 = group1.cols;
	int nGroup2 = group2.cols;
	int nMatches = matches.size();

	cv::Mat KD = K.t() + K;

	for (int iter = 0; iter < nItMa; iter++) {
		cv::Mat g;
		if (gamma > 0.0)
			g = 2 * gamma * x + (1 - gamma) * (KD * x);
		else
			g = 2 * gamma * x + (1 + gamma) * (KD * x);

		double nor = cv::norm(g, CV_L2);
		if (nor * nor < eta)
			break;

		// subproblem
	//	double maxv = cv::max(g);
	//	int n = cv::max(nGroup1, nGroup2);
	//	cv::Mat_<double> G(n, n, maxv);
	//	for (int i = 0; i < nMatches; i++)
	//		G(matches[i].idx_g1, matches[i].idx_g2) = maxv;
	//	G = KM(-G);
	//	cv::Mat_<double> y(nMatches, 1, 0.0);
	//	for (int i = 0; i < nMatches; i++)
	//		y(i) = G(matches[i].idx_g1, matches[i].idx_g2);

		// use greedy for approximation of the subproblem
		cv::Mat y = greedyMapping(g, matches);

		double errorterm = g.dot(y - x);	// sum(sum(g.*(y - x)));
		if (errorterm > -eta) {
			break;
		}
		
		x = lineSearch(y, x, K, gamma, eta);

		double Fnew = Obj_GNCCP(x, K, gamma);
		if (gamma > 0)
			g = 2 * gamma * x + (1 - gamma) * (KD * x);
		else
			g = 2 * gamma * x + (1 + gamma) * (KD * x);

		double tmp = g.dot(x - y);
		if (tmp < eta  * abs(Fnew - tmp))
			break;

	}

	return x;
}

double GMA::Obj_GNCCP(cv::Mat& x, cv::Mat& K, double gamma)
{
	cv::Mat_<double> f;
	if (gamma > 0)
		f = gamma * (x.t() * x) + (1-gamma) * (x.t() * K * x);
	else
		f = gamma * (x.t() * x) + (1+gamma) * (x.t() * K * x);
	return f(0,0);
}

cv::Mat GMA::lineSearch(cv::Mat& y, cv::Mat& x, cv::Mat& K, double gamma, double eta)
{
	cv::Mat Pright = y.clone();
	cv::Mat Pleft = x.clone();
	cv::Mat deltaX = Pright - Pleft;

	cv::Mat Pnew, Pnew2;

	for (int i = 0; i < 10; i++) {
		Pnew = Pleft + 0.5 * deltaX;
		Pnew2 = Pleft + (0.5 + eta) * deltaX;
		double F0 = Obj_GNCCP(Pnew, K, gamma);
		double F1 = Obj_GNCCP(Pnew2, K, gamma);

		if (F0 < F1)
			Pright = Pnew;
		else
			Pleft = Pnew;

		deltaX = Pright - Pleft;
	}
	x = Pnew;

	return x;
}

cv::Mat GMA::EstimateByTangent(vector<cv::Mat>& x_store, vector<int>& sLen_store, bool bRefine)
{
	// estimate next solution x according to previous K solutions
	// x_store:  [N * K] matrix
	// sLen_store : K vector of step length
	int rows = x_store[0].rows;
	cv::Mat_<double> x_shift(rows, 1, 0.0);

	// estimate shift of next solution
	int num = x_store.size();
	vector<double> weight;
	double sum_weight = 0.0;
	for (int i = 0; i < num; i++) {
		weight.push_back(i + 1);
		sum_weight += weight[i];
	}

	for (int i = 0; i < num - 1; i++) {
		cv::Mat shift = (x_store[i + 1] - x_store[i]) / (double)sLen_store[i];

		x_shift = x_shift + weight[i] * shift;
	}

	x_shift = x_shift / sum_weight;
	cv::Mat_<double>  x = x_store[num - 1] + x_shift * sLen_store[num - 1];

	if (bRefine) {
		for (int i = 0; i < rows; i++) {
			x(i, 0) = min(1.0, max(0.0, x(i, 0)));
		}
	}

	return x;
}