#include <vector>
#include <cmath>

using namespace std;


class Neuron 
{
	private:
	int inCount;
	vector<double>* inVector;
	vector<double>* weights;
	double net;
	double output;
	double sigma;
	vector<double>* delta;
	double weightOffset;

	void checkIndexWeight(int indexWeight) {
		if (!(0<=indexWeight && indexWeight < this->inCount)) 
		{
			throw "Index Weight out of bounds.";
		}
	}
	
	public:
	Neuron(int inCount, double initialWeightValue) {
		this->inCount = inCount;
		this->weightOffset = 0.0;

        inVector = nullptr;
        weights = new vector<double>();
        delta = new vector<double>();

		for (int i=0;i<this->inCount;i++) 
		{
			this->weights->push_back(initialWeightValue * (0.1 + 0.8*drand48()));
            this->delta->push_back(0.0);
		}
	}

    ~Neuron()
    {
        delete inVector;
        delete weights;
        delete delta;
    }

    int getInCount() {
        return this->inCount;
    }
	
	void setInVector(vector<double>* inVector) 
	{
		this->inVector = inVector;
	}
	
	vector<double>* getInVector() 
	{
		return this->inVector;
	}
	
	double generateOutput() 
	{
		net = weightOffset;
		for (int i=0;i<inCount;i++) 
		{
			net += weights->at(i)*inVector->at(i);
		}

		output = 1 / (1 + exp(-1 * net));
		
		return output;
	}
	
	double getWeight(int indexWeight) 
	{
		checkIndexWeight(indexWeight);
		return weights->at(indexWeight);
	}
	
	void setWeight(int indexWeight, double value) 
	{
		checkIndexWeight(indexWeight);
		weights->at(indexWeight) = value;		
	}

	vector<double>* getWeights() 
	{
		return weights;
	}

	void setWeights(vector<double>* weights) 
	{
		this->weights = weights;
	}

	double getNet() 
	{
		return net;
	}

	double getOutput() 
	{
		return output;
	}

	double getSigma() 
	{
		return sigma;
	}

	void setSigma(double sigma) 
	{
		this->sigma = sigma;
	}

	vector<double>* getDelta() 
	{
		return delta;
	}

	void setDelta(vector<double>* delta) 
	{
		this->delta = delta;
	}

	double getWeightOffset() 
	{
		return weightOffset;
	}

	void setWeightOffset(double weightOffset) 
	{
		this->weightOffset = weightOffset;
	}
};

