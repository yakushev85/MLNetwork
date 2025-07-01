#include <vector>
#include <string>
#include <iostream>
#include "ml_data.cpp"
#include "ml_neuron.cpp"

using namespace std;


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

