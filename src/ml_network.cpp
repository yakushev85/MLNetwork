#include <vector>
#include <string>
#include <iostream>
#include <cmath>

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

        inVector = new vector<double>();
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


class Layer 
{
    private: 
    vector<Neuron*>* neurons;

    public: 
    Layer(int inCount, int neuronLayerCount, double initialWeightValue) 
    {
        neurons = new vector<Neuron*>();
        for (int i=0;i<neuronLayerCount;i++) 
        {
            neurons->push_back(new Neuron(inCount, initialWeightValue));
        }
    }

    ~Layer()
    {
        delete neurons;
    }

    vector<Neuron*>* getNeurons() 
    {
        return neurons;
    }
};


class MultiNetwork 
{
    private:
    NetConfiguration* configuration;
    vector<Layer*>* layers;

    public: 
    MultiNetwork(NetConfiguration* configuration) 
    {
        this->configuration = configuration;

        vector<int>* neuronCounts = configuration->neuronCounts;
        int preIn = configuration->inCount;
        layers = new vector<Layer*>();

        for (int neuronCount : *(neuronCounts)) 
        {
            Layer* layer = new Layer(preIn, neuronCount, configuration->initialWeightValue);
            layers->push_back(layer);
            preIn = neuronCount;
        }
    }

    ~MultiNetwork()
    {
        delete layers;
    }

    
    vector<double>* execute(vector<double>* inVector) 
    {
        vector<double>* layerInVector = new vector<double>();

        for (double v : *inVector)
        {
            layerInVector->push_back(v);
        }

        for (Layer* layer : *layers) 
        {
			vector<Neuron*>* neurons = layer->getNeurons();

            for (Neuron* neuron : *neurons) 
            {
                neuron->setInVector(layerInVector);
                neuron->generateOutput();
            }

            layerInVector->clear();

            for (Neuron* neuron : *neurons) 
            {
                layerInVector->push_back(neuron->getOutput());
            }
        }

        return layerInVector;
    }

    
    void learn(bool showInfo) 
    {
        int currentIteration = 1;
        double outputError = 1000000.0;

        while (outputError > 0 && currentIteration < configuration->maxLearningIterations) 
        {
            outputError = iteration();

            if (showInfo)
            {
                cout << currentIteration << ". error = " << outputError << endl;
            }

            currentIteration++;
        }

        cout << "MultiNetwork.learn finished with error = " << outputError << " iteration = " << currentIteration << endl;
    }

    double iteration() 
    {
        double totalErrorSum = 0.0;
        vector<TeachDataEntity>* learningData = configuration->teachData;

        double alphaValue = configuration->alpha;
        double speedValue = configuration->speed;
        int outputSize = configuration->neuronCounts->at(configuration->neuronCounts->size()-1);

        for (TeachDataEntity teachData : *(learningData)) 
        {
            // step 1 execute net with teach data
            vector<double>* actualOutput = execute(teachData.inp);
            vector<double>* expectedOutput = teachData.output;

            // step 2 generate sigma for last layer and update errorSum
            for (int j=0;j<outputSize;j++) 
            {
                layers->at(layers->size()-1)->getNeurons()->at(j)->setSigma(
                    -1.0*actualOutput->at(j)*(1-actualOutput->at(j))*(expectedOutput->at(j)-actualOutput->at(j))
                );
                totalErrorSum += (abs(expectedOutput->at(j)-actualOutput->at(j)) > 0.5)? 1.0 : 0.0;
            }

            // step 3 generate sigma for other layers
            for (int i=layers->size()-2;i>=0;i--) 
            {
                for (int j=0;j<layers->at(i)->getNeurons()->size();j++) 
                {
                    double currentSigma = 0.0;
                    double output = layers->at(i)->getNeurons()->at(j)->getOutput();
                    double preSigma = output*(1-output);

                    for (Neuron* neuron : *(layers->at(i+1)->getNeurons())) 
                    {
                        currentSigma += neuron->getWeight(j) * neuron->getSigma();
                    }

                    currentSigma = preSigma * currentSigma;

                    layers->at(i)->getNeurons()->at(j)->setSigma(currentSigma);
                }
            }

            // step 4.1 generate delta
            for (int k=0;k<layers->size();k++) 
            {
                vector<double>* output = new vector<double>();
                if (k == 0) 
                {
                    for (double v : *(teachData.inp)) {
                        output->push_back(v);
                    }
                } 
                else 
                {
                    int outSize = layers->at(k-1)->getNeurons()->size();
                    output->clear();

                    for (int l=0;l<outSize;l++) 
                    {
                        output->push_back(layers->at(k-1)->getNeurons()->at(l)->getOutput());
                    }
                }

                for (int j=0;j<layers->at(k)->getNeurons()->size();j++) 
                {
                    vector<double>* delta = layers->at(k)->getNeurons()->at(j)->getDelta();

                    for (int i=0;i<delta->size();i++) 
                    {
                        double currentSigma = layers->at(k)->getNeurons()->at(j)->getSigma();
                        delta->at(i) = alphaValue*delta->at(i)+(1-alphaValue)*speedValue*currentSigma*output->at(i);
                    }
                }

                delete output;
            }

            // step 4.2 update weights
            for (Layer* layer : *(layers)) 
            {
                for (Neuron* neuron : *(layer->getNeurons())) 
                {
                    vector<double>* currentDelta = neuron->getDelta();

                    for (int i=0;i<neuron->getInCount();i++) 
                    {
                        neuron->setWeight(i, neuron->getWeight(i) - currentDelta->at(i));
                    }
                }
            }
        }

        return totalErrorSum;
    }
};

