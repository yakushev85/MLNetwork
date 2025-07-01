#include <vector>

using namespace std;

struct TeachDataEntity 
{
	vector<double>* inp;
	vector<double>* output;
};

struct NetConfiguration 
{
    int inCount;
    vector<int>* neuronCounts;
    int maxLearningIterations;
    double initialWeightValue;
    double alpha, speed;
    vector<TeachDataEntity>* teachData;
};
