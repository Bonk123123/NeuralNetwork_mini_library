#include "AI.h"

AI::AI()
{
	// при создании сразу создаётся первый слой с одним нейроном
	vector<float> layer;
	layer.resize(1);
	layer[0] = 1;

	this->Neurons.push_back(layer);
}

AI::AI(size_t quantityInputs)
{
	// создаётся слой с желаемым количеством нейронов
	vector<float> layer;

	layer.resize(quantityInputs);
	for (auto Neuron{ layer.begin() }; Neuron != layer.end(); Neuron++)
	{
		(*Neuron) = 1;
	}

	this->Neurons.push_back(layer);
}

AI::AI(size_t quantityInputs, int quantityBias)
{
	// создаётся слой с желаемым количеством нейронов и биасов
	vector<float> layer;

	layer.resize(quantityInputs + quantityBias);
	for (auto Neuron{ layer.begin() }; Neuron != layer.end(); Neuron++)
	{
		(*Neuron) = 1;
	}

	this->Neurons.push_back(layer);
}


// определение оптимайзера ( sgd, sgdnest, adagrad, rms, adadelta, adam ) если использовать эту функцию то параметры будут настроены по дефолту
void AI::setOptimizer(string optimizer)
{
	this->optimizer = optimizer;
}
//

// опреление оптимизаторов с своими значениями
void AI::setSGDoptimizer(float lr)
{
	this->lr = lr;
}
void AI::setSGDNESToptimizer(float gamma, float v0, float eps)
{
	this->gamma = gamma;
	this->v0 = v0;
	this->eps = eps;
}
void AI::setADAGRADoptimizer(float lr, float eps)
{
	this->lr = lr;
	this->eps = eps;
}
void AI::setRMSoptimizer(float lr, float gamma, float eps)
{
	this->lr = lr;
	this->gamma = gamma;
	this->eps = eps;
}
void AI::setADADELTAoptimizer(float gamma, float eps)
{
	this->gamma = gamma;
	this->eps = eps;
}
void AI::setADAMoptimizer(float lr, float b1, float b2, float m, float v, float eps)
{
	this->lr = lr;
	this->b1 = b1;
	this->b2 = b2;
	this->m = m;
	this->v = v;
	this->eps = eps;
}


// добавление нового слоя с желаемым количеством нейронов и биасов, а также нужно указать функцию активации для этого слоя
// веса генерируются равномерно от 0 до 1, при желании можно написать нормальное распределение 
void AI::addLayer(size_t count, int quantityBias, string actFun)
{
	//srand(time(0));
	vector<float> layer;
	vector<vector<float>> weights;

	layer.resize(count + quantityBias);
	for (auto Neuron{ layer.begin() }; Neuron != layer.end(); Neuron++)
	{
		(*Neuron) = 1;
	}

	
	weights.resize(count + quantityBias);

	

	for (auto Weight{ weights.begin() }; Weight != weights.end(); Weight++)
	{
		(*Weight).resize(this->Neurons.back().size());
		for (auto WeightNeuron{ (*Weight).begin() }; WeightNeuron != (*Weight).end(); WeightNeuron++)
		{
			(*WeightNeuron) = ((float)rand() / (float)RAND_MAX);
		}
	}
	this->Neurons.push_back(layer);
	this->Weights.push_back(weights);
	this->actFunLayers.push_back(actFun);
}


// функция нужная только для внутри класса, она устанавливает все входы в первый слой
void AI::setInputLayerNeurons(vector<float> data)
{
	if (data.size() > this->Neurons[0].size()) throw runtime_error("error");

	for (size_t i = 0; i < data.size(); i++)
	{
		this->Neurons[0][i] = data[i];
	}
}
//

// определение размера батча
void AI::setBatchSize(int batchSize)
{
	this->batch = batchSize;
}
//

// функция для прямого хода нейросети
vector<float> AI::predict(vector<float> data)
{
	vector<float> answer;
	
	// устанавливает все входы в первый слой
	this->setInputLayerNeurons(data);

	for (size_t i = 1; i < this->Neurons.size(); i++)
	{
		for (size_t j = 0; j < this->Neurons[i].size(); j++)
		{
			float neuron = 0;
			
			// получение следующего нейрона
			for (size_t k = 0; k < this->Weights[i - 1][j].size(); k++)
			{
				neuron += this->Neurons[i - 1][k] * this->Weights[i - 1][j][k];
			}
			//
			
			// функции активации, можно добавить при необходимости, но также нужно дописать в функции getDiffs
			if (this->actFunLayers[i - 1] == "sigmoid")
			{
				neuron = 1 / (1 + exp(-neuron));
			}
			if (this->actFunLayers[i - 1] == "relu")
			{
				if (neuron <= 0.01)
				{
					neuron *= 0.01;
				}
			}
			//
			this->Neurons[i][j] = neuron;
		}
	}

	// получение выходного слоя для возврращения фукции
	answer.resize(this->Neurons.back().size());
	for (size_t i = 0; i < this->Neurons.back().size(); i++)
	{
		answer[i] = this->Neurons.back()[i];
	}
	//

	return answer;
}
//

// обратное распрастранение ошибки, на вход даётся одно из наблюдений, на выходе выдаёт вектор градиентов
vector<vector<vector<float>>> AI::getDiffs(vector<float> predict, vector<float> correct)
{
	vector<vector<vector<float>>> diffs;
	
	vector<float> deltas;

	// получение ошибки для последнего слоя
	vector<vector<float>> outputDiffs;
	for (size_t i = 0; i < this->Neurons.back().size(); i++)
	{
		vector<float> diff;
		float delta = -1 * (1/float(predict.size())) * (correct[i] - predict[i]);

		// определение дельты для разных функций активаций, тут при добавлении такой функции, нужно дописать для неё код по такому же шаблону
		if (this->actFunLayers.back() == "sigmoid")
		{
			delta *= predict[i] * (1 - predict[i]);
		}
		if (this->actFunLayers.back() == "relu")
		{
			if (predict[i] <= 0.01)
			{
				delta *= 0.01;
			}
			else
			{
				delta *= 1;
			}
		}
		//

		// получение вектора градиета для весов одного нейрона предыдущего слоя
		deltas.push_back(delta);
		for (size_t j = 0; j < this->Neurons[this->Neurons.size() - 2].size(); j++)
		{
			diff.push_back(delta * this->Neurons[this->Neurons.size() - 2][j]);
		}
		outputDiffs.push_back(diff);
		//
	}
	//
	diffs.push_back(outputDiffs);

	vector<float> hiddenDeltas;
	for (size_t i = this->Neurons.size() - 2; i > 0; i--) // цикл слоёв
	{
		vector<vector<float>> innerDiffs;

		for (size_t j = 0; j < this->Neurons[i].size(); j++) // цикл нейронов
		{

			float delta = 0;

			// рекурсивное получение ошибки
			for (size_t k = 0; k < this->Neurons[i + 1].size(); k++)
			{
				delta += this->Weights[i][k][j] * deltas[k];
			}

			// определение дельты для разных функций активаций, тут при добавлении такой функции, нужно дописать для неё код по такому же шаблону
			if (this->actFunLayers[i - 1] == "sigmoid")
			{
				delta *= 2 * this->Neurons[i][j] * (1 - this->Neurons[i][j]);
			}
			if (this->actFunLayers[i - 1] == "relu")
			{
				if (this->Neurons[i][j] <= 0.01)
				{
					delta *= 0.01;
				}
				else
				{
					delta *= 1;
				}
			}
			//
			hiddenDeltas.push_back(delta);
			
			// получение вектора градиета для весов одного нейрона предыдущего слоя
			vector<float> diff;
			for (size_t t = 0; t < this->Neurons[i - 1].size(); t++)
			{
				diff.push_back(delta * this->Neurons[i - 1][t]);
			}
			innerDiffs.push_back(diff);
			//
		}

		deltas.clear();
		deltas = hiddenDeltas;
		hiddenDeltas.clear();
		diffs.push_back(innerDiffs);
		innerDiffs.clear();
			
	}
	
	return diffs;
}
//

void AI::train(vector<vector<float>> &data, vector<vector<float>> &correct, size_t epochs)
{
	srand((unsigned)time(NULL));
	
	// переменная для аккумулирования инерции градиентов
	float gradSum = 0;
	//

	if (this->batch == 0) this->batch = data[0].size();

	// цикл для эпох
	for (size_t e = 0; e < epochs; e++)
	{
		
		// определение loss для текущей эпохи ( Mean Squared Error )
		float loss = 0;

		for (size_t i = 0; i < correct.size(); i++)
		{
			for (size_t j = 0; j < correct[i].size(); j++)
			{
				loss += pow(correct[i][j] - this->predict(data[i])[j], 2);
			}
		}
		loss /= correct.size();
		//

		int iter = 0;

		// итерации для батчей
		for (size_t i = 0; i < int((data.size() / this->batch)); i++)
		{
			
			// определение рандомного наблюдения
			int random = rand() % int((iter+1) * this->batch);
			//

			// получение вектора ошибок
			vector<vector<vector<float>>> diffs(this->getDiffs(this->predict(data[random]), correct[random]));
			reverse(diffs.begin(), diffs.end());
			//

			// PARAMETERS

			float learning_rate = this->lr;
			float gamma = this->gamma;
			
			float eps = this->eps;

			// ADADELTA parameter
			float rms = 1;
			// gamma = 0.9

			// SGDNEST parameter
			float v0 = this->v0;;
			// gamma = 0.9
				

			// ADAM parameters
			float b1 = this->b1;
			float b2 = this->b2;
			float m = this->m;
			float v = this->v;
				
			//

			// оптимизаторы

				// SGD
			if (this->optimizer == "sgd")
			{
				for (size_t i = 0; i < this->Weights.size(); i++)
				{
					for (size_t j = 0; j < this->Weights[i].size(); j++)
					{
						for (size_t k = 0; k < this->Weights[i][j].size(); k++)
						{
							this->Weights[i][j][k] -= learning_rate * diffs[i][j][k];
						}
					}
				}

			}
				//

				// оптимизатор SGDNEST sgd nesterov
			if (this->optimizer == "sgdnest")
			{
				for (size_t i = 0; i < this->Weights.size(); i++)
				{
					for (size_t j = 0; j < this->Weights[i].size(); j++)
					{
						for (size_t k = 0; k < this->Weights[i][j].size(); k++)
						{
							v0 = (gamma * v0 + (1 - gamma) * diffs[i][j][k]);
							this->Weights[i][j][k] -= v0;

						}
					}
				}

			}
				//

				// оптимизатор ADAGRAD
			if (this->optimizer == "adagrad")
			{
				for (size_t i = 0; i < this->Weights.size(); i++)
				{
					for (size_t j = 0; j < this->Weights[i].size(); j++)
					{
						for (size_t k = 0; k < this->Weights[i][j].size(); k++)
						{
							gradSum += pow(diffs[i][j][k], 2);
							this->Weights[i][j][k] -= (learning_rate / (sqrt(gradSum + eps))) * diffs[i][j][k];
						}
					}
				}

			}
				//

				// оптимизатор RMSProp
			if (this->optimizer == "rms")
			{
				for (size_t i = 0; i < this->Weights.size(); i++)
				{
					for (size_t j = 0; j < this->Weights[i].size(); j++)
					{
						for (size_t k = 0; k < this->Weights[i][j].size(); k++)
						{

							gradSum = gradSum * gamma + (1 - gamma) * pow(diffs[i][j][k], 2);
							this->Weights[i][j][k] -= (learning_rate / (sqrt(gradSum + eps))) * diffs[i][j][k];
						}
					}
				}

			}
				//

				// оптимизатор АДАДЕЛЬТА
			if (this->optimizer == "adadelta")
			{
				for (size_t i = 0; i < this->Weights.size(); i++)
				{
					for (size_t j = 0; j < this->Weights[i].size(); j++)
					{
						for (size_t k = 0; k < this->Weights[i][j].size(); k++)
						{

							gradSum = gradSum * gamma + (1 - gamma) * pow(diffs[i][j][k], 2);
							this->Weights[i][j][k] -= (rms / (sqrt(gradSum + eps))) * diffs[i][j][k];
							rms = (sqrt(gradSum + eps));
						}
					}
				}

			}
				//

				// оптимизатор АДАМ
			if (this->optimizer == "adam")
			{
				for (size_t i = 0; i < this->Weights.size(); i++)
				{
					for (size_t j = 0; j < this->Weights[i].size(); j++)
					{
						for (size_t k = 0; k < this->Weights[i][j].size(); k++)
						{
							m = b1 * m + (1 - b1) * diffs[i][j][k];
							v = b2 * v + (1 - b2) * (pow(diffs[i][j][k], 2));
							float mdop = m / (1 - pow(b1, i * j * k + 1));
							float vdop = v / (1 - pow(b2, i * j * k + 1));
							this->Weights[i][j][k] -= (learning_rate * (mdop / (sqrt(vdop) + eps)));
							
						}
					}
				}

			}
				//

			//

			// переход к следующей итерации
			iter++;
			//
		}
		//

		// ( отдельно ) обычный градиетный спуск
		if (this->optimizer == "gd")
		{
			float learning_rate = 0.1;
			for (size_t d = 0; d < data.size(); d++)
			{
				vector<vector<vector<float>>> diffs(this->getDiffs(this->predict(data[d]), correct[d]));
				reverse(diffs.begin(), diffs.end());

				for (size_t i = 0; i < this->Weights.size(); i++)
				{
					for (size_t j = 0; j < this->Weights[i].size(); j++)
					{
						for (size_t k = 0; k < this->Weights[i][j].size(); k++)
						{
							this->Weights[i][j][k] -= learning_rate * diffs[i][j][k];
						}
					}
				}
			}
		}
		//

		// выведение эпохи и loss в консоль
		cout << "epoch: " << e + 1 << " loss: " << loss << endl; 
		//
	}
	//
}
