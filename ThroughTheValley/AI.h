#pragma once

#include <string>
#include <ctime>
#include <vector>
#include <iostream>

using namespace std;

class AI
{
private:
	string optimizer = "";
	int batch = 0;


	float lr = 0.1;
	float gamma = 0.9;
	float v0 = 1;
	float eps = 0.000000001;
	float b1 = 0.9;
	float b2 = 0.9999;
	float m = 0;
	float v = 0;

	vector<string> actFunLayers;

	vector<vector<float>> Neurons;
	vector<vector<vector<float>>> Weights;

	void setInputLayerNeurons(vector<float> data);

public:
	AI();
	AI(size_t quantityInputs);
	AI(size_t quantityInputs, int quantityBias);

	void setOptimizer(string optimizer);

	void setSGDoptimizer(float lr);
	void setSGDNESToptimizer(float gamma, float v0, float eps);
	void setADAGRADoptimizer(float lr, float eps);
	void setRMSoptimizer(float lr, float gamma, float eps);
	void setADADELTAoptimizer(float gamma, float eps);
	void setADAMoptimizer(float lr, float b1, float b2, float m, float v, float eps);

	void addLayer(size_t count, int quantityBias, string actFun);
	
	void setBatchSize(int batchSize);

	vector<float> predict(vector<float> data);
	
	vector<vector<vector<float>>> getDiffs(vector<float> data, vector<float> correct);

	void train(vector<vector<float>> &data, vector<vector<float>> &correct, size_t epochs);
};

