#include "onnlab.h"

#include <iostream>

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
#include "NNB_Linear.h"
#include "NNB_ReLU.h"
#include "NNB_m1ReLU.h"
#include "NNB_Input.h"
#include "NNB_ConstInput.h"
#include "NNB_Layer.h"
#include "NNB_LayersAggregator.h"
#include "NNB_m1h_SumHead.h"
#include "FwdBackPropGuider.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "SparceLayerStaticConnectomHolder2Mult.h"
#include "SparceLayerStaticConnectomHolderOneToOne.h"
#include "NeuronHoldingStaticLayer.h"
#include "Monotonic2FeildsProjectingAccessory.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <algorithm>


void exp_m1ReLU_svsg4() {
	std::cout << "exp_m1ReLU_svsg4" << std::endl;

	constexpr float M1_MAX_VALUE = 4.0f;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);
	std::uniform_real_distribution<float> randistributor_small(0.001f, 0.1f);
	std::uniform_int_distribution<unsigned short> randistributor_int(0, 1);

	// Input vector
	std::array<float, 7> inputs_store;

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_Input> layer_inp(7, [&](nn::NNB_Input *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Input(&inputs_store[index]);
	});

	// Output m1ReLU layer 
	nn::NeuronHoldingStaticLayer<nn::NNB_m1ReLU> layer_m1lr(11, [&](nn::NNB_m1ReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_m1ReLU(0.1f, M1_MAX_VALUE);
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_Linear> layer_m1out(11, [&](nn::NNB_Linear *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_Linear(-1.0f, M1_MAX_VALUE);
	});

	// Equvivelent ReLU layer 
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_rleq_inc(11, [&](nn::NNB_ReLU *const mem_ptr, unsigned) { // Increasing (__/)
		new(mem_ptr)nn::NNB_ReLU;
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_rleq_dec(11, [&](nn::NNB_ReLU *const mem_ptr, unsigned) { // Decreasing (\__)
		new(mem_ptr)nn::NNB_ReLU;
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_m1h_SumHeadT<true, -1.0f, M1_MAX_VALUE>> layer_smout(11, [&](nn::NNB_m1h_SumHeadT<true, -1.0f, M1_MAX_VALUE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_m1h_SumHeadT<true, -1.0f, M1_MAX_VALUE>;
	});

	nn::NNB_LayersAggregator layer_rleq({ &layer_rleq_inc, &layer_rleq_dec });

	// Connections
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection> connections_bias_m1out(&layer_bias, &layer_m1lr, [&](nn::NNB_Connection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection(from, to, -M1_MAX_VALUE);
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection> connections_bias_rleqi(&layer_bias, &layer_rleq_inc, [&](nn::NNB_Connection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection(from, to, -randistributor_small(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection> connections_bias_rleqd(&layer_bias, &layer_rleq_dec, [&](nn::NNB_Connection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection(from, to, randistributor_small(preudorandom));
	});
	// Input to
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection> connections_in_m1lr(&layer_inp, &layer_m1lr, [&](nn::NNB_Connection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection(from, to, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection> connections_in_rleqi(&layer_inp, &layer_rleq_inc, [&](nn::NNB_Connection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection(from, to, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection> connections_in_rleqd(&layer_inp, &layer_rleq_dec, [&](nn::NNB_Connection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection(from, to, -randistributor(preudorandom));
	});
	// Heads
	nn::SparceLayerStaticConnectomHolderOneToOne<nn::NNB_StraightConnection> connections_m1lr_m1out(&layer_m1lr, &layer_m1out, [&](nn::NNB_StraightConnection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_StraightConnection(from, to);
	});
	nn::SparceLayerStaticConnectomHolder2Mult<nn::NNB_StraightConnection> connections_rleq_smout(&layer_rleq, &layer_smout, [&](nn::NNB_StraightConnection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_StraightConnection(from, to);
	});

	static auto projector_spread = [](float w1, float w2)->float {
		if (w1 > w2) {
			return (w1 - w2) * ((static_cast<int8_t>(w1 * w2 > 0) << 1) - 1);
		} else {
			return (w2 - w1) * ((static_cast<int8_t>(w1 * w2 > 0) << 1) - 1);
		}
	};
	auto projector_2to1 = [&](nn::interfaces::NBI *const m1, const nn::interfaces::NBI *const eq1, const nn::interfaces::NBI *const eq2) {
		auto &eq1conn = const_cast<nn::interfaces::NBI *>(eq1)->InputConnections();
		auto &eq2conn = const_cast<nn::interfaces::NBI *>(eq2)->InputConnections();
		auto &m1conn = m1->InputConnections();

		float w1 = eq1conn[0]->Weight(), w2 = eq2conn[0]->Weight();
		bool rotflag = w1 > w2; // Selecting neuron to flip: suppose, lower module of bias is better

		m1conn[0]->Weight((rotflag ? w1 - w2 : w2 - w1) / 2.0f - M1_MAX_VALUE); // Setting [correcting] bias offset
		for (unsigned i = 1; i < eq1conn.size(); ++i) { // Skipping bias
			w1 = eq1conn[i]->Weight();
			w2 = eq2conn[i]->Weight();
			if (projector_spread(w1, w2) > -1e-3f) { // Same signs (or very small module), suppose single feild usage
				m1conn[i]->Weight((rotflag ? w1 - w2 : w2 - w1) / 2.0f); // m1conn[i]->Weight((rotflag ? w1 + -w2 : -w1 + w2) / 2.0f);
			} else { // Different signs, suppose possible both feilds usage
				// Turning into positive to direct to illinear part (instead of just linear)
				m1conn[i]->Weight((std::fabs(w1) + std::fabs(w2)) / 2.0f); //m1conn[i]->Weight((rotflag ? std::fabs(w1) + std::fabs(-w2) : std::fabs(-w1) + std::fabs(w2)) / 2.0f);
			}
		}
	};

	auto projector_1to2 = [&](const nn::interfaces::NBI *const m1, nn::interfaces::NBI *const eq1, nn::interfaces::NBI *const eq2) {
		auto &eq1conn = eq1->InputConnections();
		auto &eq2conn = eq2->InputConnections();
		auto &m1conn = const_cast<nn::interfaces::NBI *>(m1)->InputConnections();

		std::vector<std::tuple<float, bool, bool>> m1proj;
		m1proj.reserve(m1conn.size());

		float w1 = eq1conn[0]->Weight(), w2 = eq2conn[0]->Weight();
		bool rotflag = w1 > w2; // Selecting neuron to flip: suppose, lower module of bias is better
		m1proj.emplace_back((rotflag ? w1 - w2 : w2 - w1) / 2.0f - M1_MAX_VALUE, false, false); // Setting [correcting] bias offset

		for (unsigned i = 1; i < eq1conn.size(); ++i) { // Skipping bias
			w1 = eq1conn[i]->Weight();
			w2 = eq2conn[i]->Weight();
			if (projector_spread(w1, w2) > -1e-3f) { // Same signs, suppose single feild usage
				m1proj.emplace_back((rotflag ? w1 - w2 : w2 - w1) / 2.0f, false, false); // m1conn[i]->Weight((rotflag ? w1 + -w2 : -w1 + w2) / 2.0f);
			} else { // Different signs, suppose possible both feilds usage
				// Turning into positive to direct to illinear part (instead of just linear)
				m1proj.emplace_back((std::fabs(w1) + std::fabs(w2)) / 2.0f, w1 > 0, w2 > 0); //m1conn[i]->Weight((rotflag ? std::fabs(w1) + std::fabs(-w2) : std::fabs(-w1) + std::fabs(w2)) / 2.0f);
			}
		}

		float rotsign = ((static_cast<int8_t>(rotflag) << 1) - 1);
		float diff;
		// Bias handling
		{
			w1 = std::get<0>(m1proj[0]);
			w2 = m1conn[0]->Weight();
			diff = std::fabs(w2 - w1); // New - Old
			if (projector_spread(w1, w2) > -1e-2f) { // Suppose bias meaning does not changed
				eq1conn[0]->Weight(eq1conn[0]->Weight() + rotsign * diff);
				eq2conn[0]->Weight(eq2conn[0]->Weight() - rotsign * diff); // eq2conn[0]->Weight() + -rotsign * diff
			} else { // Bias became to be something strange... apparently, the network has decided to use just second feild... unwanted result.
				// Regenerating biases! Serious intervention, possible convergence problems.
				eq1conn[0]->Weight(rotsign * (w2 + M1_MAX_VALUE));
				eq2conn[0]->Weight(rotsign * (M1_MAX_VALUE - w2)); // eq2conn[0]->Weight(-rotsign * (w2 - M1_MAX_VALUE))
			}
		}
		for (unsigned i = 1; i < eq1conn.size(); ++i) { // Skipping bias
			w1 = std::get<0>(m1proj[i]);
			w2 = m1conn[i]->Weight();
			diff = std::fabs(w2 - w1); // New - Old
			if (projector_spread(w1, w2) > -1e-2f) { // Suppose bias meaning does not changed
				if (std::get<1>(m1proj[i]) == std::get<2>(m1proj[i])) { // Single feild user
					eq1conn[i]->Weight(eq1conn[i]->Weight() + rotsign * diff);
					eq2conn[i]->Weight(eq2conn[i]->Weight() - rotsign * diff); // eq2conn[i]->Weight() + -rotsign * diff
				} else { // Both feilds user
					eq1conn[i]->Weight(eq1conn[i]->Weight() + ((static_cast<int8_t>(std::get<1>(m1proj[i])) << 1) - 1) * diff);
					eq2conn[i]->Weight(eq2conn[i]->Weight() + ((static_cast<int8_t>(std::get<2>(m1proj[i])) << 1) - 1) * diff);
				}
			} else { // Weight meaning [maybe] changed
				if (std::get<1>(m1proj[i]) == std::get<2>(m1proj[i])) { // Single feild user
					// Swapping the feilds
					eq1conn[i]->Weight(eq1conn[i]->Weight() - rotsign * diff);
					eq2conn[i]->Weight(eq2conn[i]->Weight() + rotsign * diff);
				} else { // Both feilds user
					// Swapping the feilds
					eq1conn[i]->Weight(eq1conn[i]->Weight() - ((static_cast<int8_t>(std::get<1>(m1proj[i])) << 1) - 1) * diff);
					eq2conn[i]->Weight(eq2conn[i]->Weight() - ((static_cast<int8_t>(std::get<2>(m1proj[i])) << 1) - 1) * diff);
				}
			}
		}
	};

	nn::utils::Monotonic2FeildsProjectingAccessory projector_rl_m1(&layer_m1lr, &layer_rleq, projector_2to1, projector_1to2);

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

	nn::BasicOutsErrorSetter::ErrorCalcSoftMAX softmax_calculator;
	nn::BasicOutsErrorSetter nerr_setter(&softmax_calculator, 11);
	nn::FwdBackPropGuider learnguider_rl({ &layer_inp, &layer_rleq, &layer_smout }, &nerr_setter);

	nn::FwdBackPropGuider learnguider_m1({ &layer_inp, &layer_m1lr, &layer_m1out }, &nerr_setter);

	std::uniform_int_distribution<unsigned> testselector(0, traindata.size() - 1);

	bool stand_learn = true;

	for (size_t iterations = 0; iterations < 4000; ++iterations) {
		// Select datarow
		const datarow *row = nullptr;
		//row = &traindata[testselector(preudorandom)];
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
		if (stand_learn) {
			learnguider_rl.DoForward();
			learnguider_rl.FillupOutsError(perfect_out);
			learnguider_rl.DoBackward();
		} else {
			learnguider_m1.DoForward();
			learnguider_m1.FillupOutsError(perfect_out);
			learnguider_m1.DoBackward();
		}

		if (iterations + 1 == 1300) {
			projector_rl_m1.Perform2to1LossyCompression();
			stand_learn = false;
		} /*else if (iterations == 2000) {
			projector_rl_m1.Perform1to2DiffTransfer();
			stand_learn = true;
		}*/
	}

	// Fine tunning
	for (size_t iterations = 0; iterations < 700; ++iterations) { // 20 iterations are enough for 91% accuracy, rest iteration to train datarow({5,6}, {1})
		// Select datarow
		const datarow *row = &traindata[testselector(preudorandom)];

		// Update inputs
		std::copy(row->inputs.begin(), row->inputs.end(), inputs_store.begin());
		const auto &perfect_out = row->outputs;

		learnguider_m1.DoForward();
		learnguider_m1.FillupOutsError(perfect_out);
		learnguider_m1.DoBackward();
	}

	// Non-monotonity usage stats: minus_count, plus_count, min, summ, max
	std::map<const nn::NNB_m1ReLU *, std::tuple<unsigned, unsigned, float, float, float>> fu_both_neurons;

	auto NonMonotonityStatProc = [&](nn::NNB_m1ReLU &nrn) {
		auto iter = fu_both_neurons.find(&nrn);
		if (iter != fu_both_neurons.end()) {
			if (nrn.OwnAccumulatorValue() < 0) {
				++std::get<0>(iter->second);
			} else {
				++std::get<1>(iter->second);
			}
			std::get<3>(iter->second) += nrn.OwnAccumulatorValue();
			if (nrn.OwnAccumulatorValue() < std::get<2>(iter->second)) {
				std::get<2>(iter->second) = nrn.OwnAccumulatorValue();
			} else if (nrn.OwnAccumulatorValue() > std::get<4>(iter->second)) {
				std::get<4>(iter->second) = nrn.OwnAccumulatorValue();
			}
		} else {
			bool minus = nrn.OwnAccumulatorValue() < 0;
			fu_both_neurons.emplace(&nrn, std::make_tuple((unsigned)(minus), (unsigned)(!minus), nrn.OwnAccumulatorValue(), nrn.OwnAccumulatorValue(), nrn.OwnAccumulatorValue()));
		}
	};
	// Inferencing
	std::cout << "learnguider_rl" << std::endl;

	for (const auto &sample : traindata) {
		// Update inputs
		std::copy(sample.inputs.begin(), sample.inputs.end(), inputs_store.begin());

		// Inference
		learnguider_rl.DoForward();

		float max = std::numeric_limits<float>::min(), max2 = max;
		int idx = -1, idx2 = -1;
		for (unsigned i = 0, cnt = layer_smout.Neurons().size(); i != cnt; ++i) {
			float value = layer_smout.Neurons()[i]->OwnLevel();
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
		std::cout << perfect_ans << " | " << idx << '\t' << max << " | " << idx2 << '\t' << max2 << std::endl;
	}

	std::cout << std::endl << "learnguider_m1" << std::endl;

	std::vector<std::tuple<unsigned, std::tuple<int, float>, std::tuple<int, float>>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		std::copy(sample.inputs.begin(), sample.inputs.end(), inputs_store.begin());

		// Inference
		learnguider_m1.DoForward();

		float max = std::numeric_limits<float>::min(), max2 = max;
		int idx = -1, idx2 = -1;
		for (unsigned i = 0, cnt = layer_m1out.Neurons().size(); i != cnt; ++i) {
			float value = layer_m1out.Neurons()[i]->OwnLevel();
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
		for (auto &nrn : layer_m1lr.NeuronsInside()) {
			NonMonotonityStatProc(nrn);
		}
	}

	for (auto &iter : fu_both_neurons) {
		std::get<3>(iter.second) /= (std::get<0>(iter.second) + std::get<1>(iter.second)) * M1_MAX_VALUE; // Normalizing average
		std::get<2>(iter.second) /= M1_MAX_VALUE; // Normalizing miniumm
		std::get<4>(iter.second) /= M1_MAX_VALUE; // Normalizing maximum
	}

	for (const auto &tpl : results) {
		std::cout << std::get<0>(tpl) << " | " << std::get<0>(std::get<1>(tpl)) << '\t' << std::get<1>(std::get<1>(tpl)) << " | " << std::get<0>(std::get<2>(tpl)) << '\t' << std::get<1>(std::get<2>(tpl)) << std::endl;
	}

	return;
}