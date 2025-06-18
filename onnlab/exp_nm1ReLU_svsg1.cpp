#include "onnlab.h"

#include "NNB_Connection_spyable.h"
#include "OptimizerGD.h"
#include "NNB_nm1ReLU.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "NNB_ConstInput.h"
#include "LearnGuiderFwBPg.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "NeuronHoldingStaticLayer.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <algorithm>

#include <iostream>


void exp_nm1ReLU_svsg1() {
	std::cout << "exp_nm1ReLU_svsg1" << std::endl;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.0f, 1.0f);
	std::uniform_int_distribution<unsigned short> randistributor_int(0, 1);

	using nnNmReLU = nn::NNB_nm1ReLUb<1, false, true>;

	// Input vector
	std::array<float, 7> inputs_store;

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput();
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_Input> layer_inp(7, [&](nn::NNB_Input *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Input(&inputs_store[index]);
	});

	// Hidden layer 1
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer_relu1(5, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(10.0f, 0.1f);
	});

	// Output layer
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer_out(11, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(10.0f, 0.1f);
	});

	// Connections
	using OptimGD = nn::optimizers::GradientDescendent;
	OptimGD optimizerGD;

	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_bias_lr1(&layer_bias, &layer_relu1, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, -10.0f);
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_bias_out(&layer_bias, &layer_out, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, -10.0f);
	});

	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_inp_lr1(&layer_inp, &layer_relu1, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, randistributor(preudorandom));
	});

	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_lr1_out(&layer_relu1, &layer_out, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, randistributor(preudorandom));
	});

	// Train data
	/* Shape is 7-segment digit input. Task: recognize digit.
	*  61112
	*  6   2
	*  00000
	*  5   3
	*  54443
	*/
	struct datarow {
		datarow(std::initializer_list<unsigned> inputs_nonzero_idx, std::initializer_list<unsigned> outputs_nonzero_idx) {
			inputs.fill(0.0f); outputs.resize(11, 0.0f);
			for (auto idx : inputs_nonzero_idx) inputs[idx] = 1.0f;
			for (auto idx : outputs_nonzero_idx) outputs[idx] = 1.0f;
		}
		std::array<float, 7> inputs;
		std::vector<float> outputs;
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

	datarow wrong_row({}, {10});

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
	nn::LearnGuiderFwBPg learnguider({ &layer_inp, &layer_relu1, &layer_out}, &softmax_calculator);

	std::uniform_int_distribution<unsigned> testselector(0, traindata.size() - 1);

	std::map<const datarow *, std::vector<nnNmReLU *>> flippers;

	for (size_t iterations = 0; iterations < 4000; ++iterations) {
		// Select datarow
		const datarow *row = nullptr;
		if (randistributor_int(preudorandom)) {
			row = &traindata[testselector(preudorandom)];
		} else {
			FillUpWrongRow();
			row = &wrong_row;
		}

		// Update inputs
		std::copy(row->inputs.begin(), row->inputs.end(), inputs_store.begin());
		const auto &perfect_out = row->outputs;

		// Learning
		learnguider.DoForward();
		learnguider.FillupOutsError(perfect_out);
		learnguider.DoBackward();

		if (iterations < 300 && (iterations + 1) % 3 == 0) {
			for (auto &nrn : layer_relu1.NeuronsInside()) {
				nrn.BatchAnalyzer_Reset();
				nrn.BatchAnalyzer_SetDataPayloadPtrSource(reinterpret_cast<const void**>(&row));
				nrn.BatchAnalyzer_SetState(true);
			}
			for (auto &nrn : layer_out.NeuronsInside()) {
				nrn.BatchAnalyzer_Reset();
				nrn.BatchAnalyzer_SetDataPayloadPtrSource(reinterpret_cast<const void **>(&row));
				nrn.BatchAnalyzer_SetState(true);
			}

			for (auto &itm : traindata) {
				std::copy(row->inputs.begin(), row->inputs.end(), inputs_store.begin());
				learnguider.DoForward();
			}

			for (auto &nrn : layer_relu1.NeuronsInside()) {
				nrn.BatchAnalyzer_SetState(false);
				auto &ref = nrn.BatchAnalyzer_GetFieldsActivateMinDistance();
				float distance = std::max(ref[0], ref[1]);
				if (distance > 1e-7f && distance < 0.5f) { // The field has not met && distance to another field not so big
					const datarow *row = reinterpret_cast<const datarow *>(nrn.BatchAnalyzer_GetFieldsActivateMinDistanceDropOnPayload()[ref[0] < ref[1]].first);
					flippers[row].push_back(&nrn);
				}
			}
			for (auto &nrn : layer_out.NeuronsInside()) {
				nrn.BatchAnalyzer_SetState(false);
				auto &ref = nrn.BatchAnalyzer_GetFieldsActivateMinDistance();
				float distance = std::max(ref[0], ref[1]);
				if (distance > 1e-7f && distance < 0.5f) { // The field has not met && distance to another field not so big
					const datarow *row = reinterpret_cast<const datarow *>(nrn.BatchAnalyzer_GetFieldsActivateMinDistanceDropOnPayload()[ref[0] < ref[1]].first);
					flippers[row].push_back(&nrn);
				}
			}

			for (auto &set : flippers) {
				std::copy(set.first->inputs.begin(), set.first->inputs.end(), inputs_store.begin());
				learnguider.DoForward();
				for (auto nrn : set.second) {
					nrn->FlipCurrentInputToAnotherField(); // Moving current input processing to another part of activaiton function
				}
			}
		}
	}

	// Non-monotonity usage stats: minus_count, plus_count, min, summ, max
	std::map<const nnNmReLU *, std::tuple<unsigned, unsigned, float, float, float>> fu_both_neurons;

	auto NonMonotonityStatProc = [&](nnNmReLU &nrn) {
		auto iter = fu_both_neurons.find(&nrn);
		if (iter != fu_both_neurons.end()) {
			if (nrn.RealAccumulatorValue() < 0) {
				++std::get<0>(iter->second);
			} else {
				++std::get<1>(iter->second);
			}
			std::get<3>(iter->second) += nrn.RealAccumulatorValue();
			if (nrn.RealAccumulatorValue() < std::get<2>(iter->second)) {
				std::get<2>(iter->second) = nrn.RealAccumulatorValue();
			} else if (nrn.RealAccumulatorValue() > std::get<4>(iter->second)) {
				std::get<4>(iter->second) = nrn.RealAccumulatorValue();
			}
		} else {
			bool minus = nrn.RealAccumulatorValue() < 0;
			fu_both_neurons.emplace(&nrn, std::make_tuple((unsigned)(minus), (unsigned)(!minus), nrn.RealAccumulatorValue(), nrn.RealAccumulatorValue(), nrn.RealAccumulatorValue()));
		}
	};

	// Inferencing
	std::vector<std::tuple<unsigned, std::tuple<unsigned, float>, std::tuple<unsigned, float>>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		std::copy(sample.inputs.begin(), sample.inputs.end(), inputs_store.begin());

		// Inference
		learnguider.DoForward();

		float max = -std::numeric_limits<float>::infinity(), max2 = max;
		unsigned idx = 0, idx2 = 0;
		for (unsigned i = 0, cnt = layer_out.Neurons().size(); i != cnt; ++i) {
			float value = layer_out.Neurons()[i]->OwnLevel();
			if (max < value) {
				max2 = max;
				idx2 = idx;
				max = value;
				idx = i;
			} else if (max2 < value) {
				max2 = value;
				idx2 = i;
			}
		}
		unsigned perfect_ans = 0;
		for (unsigned i = 0; i != sample.outputs.size(); ++i) {
			float value = sample.outputs[i];
			if (value > 0.5f) {
				perfect_ans = i;
				break;
			}
		}
		results.push_back(std::make_tuple(perfect_ans, std::make_tuple(idx, max), std::make_tuple(idx2, max2)));

		// Grab non-monotonity stat
		for (auto &nrn : layer_relu1.NeuronsInside()) {
			NonMonotonityStatProc(nrn);
		}
		for (auto &nrn : layer_out.NeuronsInside()) {
			NonMonotonityStatProc(nrn);
		}
	}

	for (auto &iter : fu_both_neurons) {
		std::get<3>(iter.second) /= std::get<0>(iter.second) + std::get<1>(iter.second);
	}

	for (const auto &tpl : results) {
		std::cout << std::get<0>(tpl) << " | " << std::get<0>(std::get<1>(tpl)) << '\t' << std::get<1>(std::get<1>(tpl)) << " | " << std::get<0>(std::get<2>(tpl)) << '\t' << std::get<1>(std::get<2>(tpl)) << std::endl;
	}

	return;
}