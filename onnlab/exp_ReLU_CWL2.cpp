#include "onnlab.h"

#include "NNB_Connection_spyable.h"
#include "OptimizerAdam.h"
#include "OptimizerGD.h"
#include "NNB_ReLU.h"
#include "NNB_Input_spyable.h"
#include "NNB_Layer.h"
#include "LearnGuiderFwBPg.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "NeuronHoldingStaticLayer.h"
#include "IterableAggregation.h"

//#include "CarliniWagnerL2.h"

#include <vector>
#include <random>
#include <array>
#include <tuple>

#include <iostream>

static std::vector<float> Learn_ReLU_svsg2();
static bool Validate_ReLU_svsg2(nn::LearnGuiderFwBPg &learnguider);

void exp_ReLU_CWL2() {
	std::cout << "exp_ReLU_CWL2" << std::endl;

	auto weights = Learn_ReLU_svsg2();

	const unsigned ATTACK_BATCH_SIZE = 4;

	// Input vector
	std::array<std::array<float, 7>, ATTACK_BATCH_SIZE> inputs_store;
	for (auto &arr : inputs_store) {
		arr.fill(0.0f);
	}

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_Input_spyableB<ATTACK_BATCH_SIZE>> layer_inp(7, [&](nn::NNB_Input_spyableB<ATTACK_BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Input_spyableB<ATTACK_BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});
	// Hidden layer 1
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<ATTACK_BATCH_SIZE>> layer_relu1(6, [&](nn::NNB_ReLUb<ATTACK_BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<ATTACK_BATCH_SIZE>;
	});
	// Output layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<ATTACK_BATCH_SIZE>> layer_out(11, [&](nn::NNB_ReLUb<ATTACK_BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<ATTACK_BATCH_SIZE>;
	});
	// Connections
	using NoOptim = nn::optimizers::GradientDescendent;
	NoOptim optimizer(0.0f); // Disable learning
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<NoOptim>> connections_inp_lr1(&layer_inp, &layer_relu1, [&](nn::NNB_Connection<NoOptim> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<NoOptim>(from, to, &optimizer);
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<NoOptim>> connections_lr1_out(&layer_relu1, &layer_out, [&](nn::NNB_Connection<NoOptim> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<NoOptim>(from, to, &optimizer);
	});

	{ // Restoring weights
		using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<NoOptim>>::CBIterator;
		IterableAggregation<nn::interfaces::CBI *> weights_iter;
		weights_iter.AppendIterableItem<IterType>(connections_inp_lr1);
		weights_iter.AppendIterableItem<IterType>(connections_lr1_out);

		std::reverse(weights.begin(), weights.end());
		for (auto ifc : weights_iter) {
			ifc->Weight(weights.back());
			weights.pop_back();
		}
	}

	nn::errcalc::ErrorCalcSoftMAX softmax_calculator(11);
	nn::LearnGuiderFwBPg learnguider({ &layer_inp, &layer_relu1, &layer_out }, &softmax_calculator, ATTACK_BATCH_SIZE);

	if (Validate_ReLU_svsg2(learnguider)) {
		std::cout << "Ready.\n";
	} else {
		std::cout << "Error!\n";
		return;
	}

	std::vector<float> outputs(11, 0.5f);
	learnguider.FillupOutsError(outputs, 0);
	learnguider.FillupOutsError(outputs, 1);
	learnguider.FillupOutsError(outputs, 2);
	learnguider.FillupOutsError(outputs, 3);
	learnguider.DoBackward();

	for (auto &nrn : layer_inp.NeuronsInside()) {
		nrn.BackPropResetError();
	}

	return;
}


static bool Validate_ReLU_svsg2(nn::LearnGuiderFwBPg &learnguider) {
	struct datarow {
		datarow(std::initializer_list<unsigned> inputs_nonzero_idx, std::initializer_list<unsigned> outputs_nonzero_idx) {
			inputs.fill(0.0f);
			outputs.fill(0.0f);
			for (auto idx : inputs_nonzero_idx) inputs[idx] = 1.0f;
			for (auto idx : outputs_nonzero_idx) outputs[idx] = 1.0f;
		}
		std::array<float, 7> inputs;
		std::array<float, 11> outputs;
	};
	std::vector<datarow> traindata = {
		datarow({1,2,3,4,5,6}, {0}),
		datarow({2,3}, {1}), datarow({5,6}, {1}),
		datarow({1,2,0,5,4}, {2}),
		datarow({1,2,3,0,4}, {3}),
		datarow({6,0,2,3}, {4}),
		datarow({1,6,0,3,4}, {5}),
		datarow({0,1,3,4,5,6}, {6}),
		datarow({1,2,3}, {7}),
		datarow({0,1,2,3,4,5,6}, {8}),
		datarow({0,1,2,3,4,6}, {9})
	};

	auto &inputs = learnguider.GetLayers()[0]->Neurons();

	std::vector<std::tuple<unsigned, std::tuple<unsigned, float>, std::tuple<unsigned, float>>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		for (size_t idx = 0; idx != sample.inputs.size(); ++idx) {
			dynamic_cast<nn::interfaces::InputNeuronI *>(inputs[idx])->SetOwnLevel(sample.inputs[idx]);
		}
		// Inference
		learnguider.DoForward();

		float max = std::numeric_limits<float>::min();
		int idx = -1;
		for (int i = 0, cnt = learnguider.GetOutputs().size(); i != cnt; ++i) {
			float value = learnguider.GetOutputs()[i]->OwnLevel();
			if (max < value) {
				max = value;
				idx = i;
			}
		}
		int perfect_ans = 0;
		for (int i = 0; i != sample.outputs.size(); ++i) {
			float value = sample.outputs[i];
			if (value > 0.5f) {
				perfect_ans = i;
				break;
			}
		}
		if (perfect_ans != idx) {
			return false;
		}
	}
	return true;
}

static std::vector<float> Learn_ReLU_svsg2() {
	std::vector<float> weights; // For Return-Value-Optimization

	const unsigned BATCH_SIZE = 4;

	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.0f, 1.0f);
	std::uniform_int_distribution<unsigned short> randistributor_int(0, 1);

	// Input vector
	std::array<std::array<float, 7>, BATCH_SIZE> inputs_store;
	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(7, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});
	// Hidden layer 1
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> layer_relu1(6, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});
	// Output layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> layer_out(11, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});
	// Connections
	using OptimAdam = nn::optimizers::Adam;
	OptimAdam optimizer(0.025f);
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAdam>> connections_inp_lr1(&layer_inp, &layer_relu1, [&](nn::NNB_Connection<OptimAdam> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAdam>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAdam>> connections_lr1_out(&layer_relu1, &layer_out, [&](nn::NNB_Connection<OptimAdam> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAdam>(from, to, &optimizer, randistributor(preudorandom));
	});

	struct datarow {
		datarow(std::initializer_list<unsigned> inputs_nonzero_idx, std::initializer_list<unsigned> outputs_nonzero_idx) {
			inputs.fill(0.0f);
			outputs.fill(0.0f);
			for (auto idx : inputs_nonzero_idx) inputs[idx] = 1.0f;
			for (auto idx : outputs_nonzero_idx) outputs[idx] = 1.0f;
		}
		std::array<float, 7> inputs;
		std::array<float, 11> outputs;
	};
	std::vector<datarow> traindata = {
		datarow({1,2,3,4,5,6}, {0}),
		datarow({2,3}, {1}), datarow({5,6}, {1}),
		datarow({1,2,0,5,4}, {2}),
		datarow({1,2,3,0,4}, {3}),
		datarow({6,0,2,3}, {4}),
		datarow({1,6,0,3,4}, {5}),
		datarow({0,1,3,4,5,6}, {6}),
		datarow({1,2,3}, {7}),
		datarow({0,1,2,3,4,5,6}, {8}),
		datarow({0,1,2,3,4,6}, {9})
	};

	datarow wrong_row({}, { 10 });
	auto FillUpWrongRow = [&]() {
		bool is_train;
		do {
			for (auto &itm : wrong_row.inputs) {
				itm = randistributor_int(preudorandom);
			}
			is_train = false;
			for (const auto &row : traindata) {
				if (row.inputs == wrong_row.inputs) {
					is_train = true;
					break;
				}
			}
		} while (is_train);
	};

	nn::errcalc::ErrorCalcSoftMAX softmax_calculator(11);
	nn::LearnGuiderFwBPg learnguider({ &layer_inp, &layer_relu1, &layer_out }, &softmax_calculator, BATCH_SIZE);

	std::uniform_int_distribution<unsigned> testselector(0, traindata.size() - 1);

	std::array<std::vector<float>, BATCH_SIZE> perfect_out_store;
	perfect_out_store.fill(std::vector<float>(11));

	size_t OVERALL_ITERATIONS = 5000;
	for (size_t iterations = 0, iters = OVERALL_ITERATIONS / BATCH_SIZE; iterations < iters; ++iterations) {
		const datarow *row = nullptr;
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			if (randistributor_int(preudorandom)) {
				row = &traindata[testselector(preudorandom)];
			} else {
				FillUpWrongRow();
				row = &wrong_row;
			}
			std::copy(row->inputs.begin(), row->inputs.end(), inputs_store[batch_i].begin());
			std::copy(row->outputs.begin(), row->outputs.end(), perfect_out_store[batch_i].begin());
		}
		learnguider.DoForward();
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			learnguider.FillupOutsError(perfect_out_store[batch_i], batch_i);
		}
		learnguider.DoBackward();
	}

	using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAdam>>::CBIterator;

	// Storing connections weights
	IterableAggregation<nn::interfaces::CBI *> weights_iter;
	weights_iter.AppendIterableItem<IterType>(connections_inp_lr1);
	weights_iter.AppendIterableItem<IterType>(connections_lr1_out);

	for (auto ifc : weights_iter) {
		weights.push_back(ifc->Weight());
	}
	return weights;
}