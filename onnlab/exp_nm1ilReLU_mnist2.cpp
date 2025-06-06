#include "onnlab.h"

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
#include "OptimizerGD.h"
#include "OptimizerAdam.h"
#include "NNB_LinearSlim.h"
#include "NNB_ilReLU.h"
#include "NNB_nm1iReLU.h"
#include "NNB_Input.h"
#include "NNB_ConstInput.h"
#include "NNB_Layer.h"
#include "NNB_LayersAggregator.h"
#include "NNB_ConvolutionHead.h"
#include "NNB_ConvolutionEssence1d.h"
#include "LearnGuiderFwBPgThreadAble.h"
#include "NeuronHoldingStaticLayer.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "SparceLayerStaticConnectomHolder2Mult.h"
#include "Monotonic2FieldsProjectingAccessory.h"
#include "Monotonic2FieldsHeuristicsEqExV2.h"

#include "CSVreader.h"
#include "NetQualityCalcUtils.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <algorithm>
#include <thread>

#include <iostream>
#include <iomanip>


void exp_nm1ilReLU_mnist2() {
	std::cout << "exp_nm1ilReLU_mnist2" << std::endl;

	const unsigned THREADS_COUNT = 32;
	const unsigned LOG_INTERVAL = 20;

	const unsigned BATCH_SIZE = 64;
	const unsigned EPOCHS = 10;
	const float VALID_TEST_SPLIT_RATIO = 0.1f; // sizeof(validation)/sizeof(full testing dataset size), should be small to run fast

	const float LEARNING_RATE = 0.001f;

	const float NM1_MAX_VALUE = 8.0f;
	const float PROJECTION_THRESHOLD = -0.4f;

	const unsigned LAYER_HIDDEN1_SIZE = 32;
	const unsigned LAYER_HIDDEN2_SIZE = 16;

	const unsigned RANDOM_SEED = 42;

	struct MNIST_entry {
		MNIST_entry(uint8_t label, std::vector<uint8_t> input, uint16_t reseve):label(label), input(input) {
			input.reserve(reseve);
		}
		uint8_t label;
		std::vector<uint8_t> input;
	};

	std::cout << "Loading train dataset...\n";

	std::vector<MNIST_entry> mnist_train;
	{
		CSVreader csv("../mnist-in-csv/mnist_train.csv");

		auto rowsz = csv.FetchNextRow();
		while (rowsz) {
			mnist_train.emplace_back(atoi(csv[0]), std::vector<uint8_t>(), rowsz - 1); // label writing

			for (unsigned i = 1; i < rowsz; ++i) {
				mnist_train.back().input.push_back(atoi(csv[i]));
			}
			rowsz = csv.FetchNextRow();
		}
	}
	nn::netquality::ClassWeightsCalculator mnist_train_classes_w(10);
	for (auto &itm : mnist_train) {
		mnist_train_classes_w.NoteSample(itm.label);
	}
	mnist_train_classes_w.CalcWeights();
	mnist_train_classes_w.KeepWeightsOnly();

	std::cout << "Loading test dataset (and valid/test split)...\n";

	std::vector<MNIST_entry> mnist_test;
	{
		CSVreader csv("../mnist-in-csv/mnist_test.csv");

		auto rowsz = csv.FetchNextRow();
		while (rowsz) {
			mnist_test.emplace_back(atoi(csv[0]), std::vector<uint8_t>(), rowsz - 1); // label writing

			for (unsigned i = 1; i < rowsz; ++i) {
				mnist_test.back().input.push_back(atoi(csv[i]));
			}
			rowsz = csv.FetchNextRow();
		}
	}
	nn::netquality::ClassWeightsCalculator mnist_test_classes_w(10);
	for (auto &itm : mnist_test) {
		mnist_test_classes_w.NoteSample(itm.label);
	}
	mnist_test_classes_w.CalcWeights();
	mnist_test_classes_w.KeepWeightsOnly();

	//std::random_device randevice;
	std::mt19937 preudorandom(RANDOM_SEED);

	std::shuffle(mnist_test.begin(), mnist_test.end(), preudorandom);
	std::vector<MNIST_entry> mnist_valid;
	mnist_valid.reserve(static_cast<unsigned>(VALID_TEST_SPLIT_RATIO * mnist_test.size()));
	{
		std::copy_n(mnist_test.begin(), mnist_valid.capacity(), std::back_inserter(mnist_valid));
	}

	std::uniform_real_distribution<float> randistributor(0.01f, 0.1f);
	std::uniform_real_distribution<float> randistributor_diviation(-0.0025f, 0.0025f);

	const unsigned LAYER_INPUT_SIZE = 28 * 28;

	using nnEqReLU = nn::NNB_ilReLUb<BATCH_SIZE, true>; // Immortal Limited ReLU, using Kahan summation
	using nnNmReLU = nn::NNB_nm1iReLUb<BATCH_SIZE, true, true>; // Non-monotonic Immortal ReLU, using Kahan summation
	using nnLinSumm = nn::NNB_LinearSlimT<BATCH_SIZE, true>; // Linear summator, using Kahan


	// Input image (like vector, flatten)
	std::vector<std::array<float, LAYER_INPUT_SIZE>> inputs_store(BATCH_SIZE);

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(LAYER_INPUT_SIZE, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});

	// Non-monotonic Hidden layer 1
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer1_nm_hid(LAYER_HIDDEN1_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});
	// Non-monotonic Hidden layer 2
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer2_nm_hid(LAYER_HIDDEN2_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});
	// Non-monotonic Output layer
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer3_nm_out(10, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});


	// Equivalent Hidden layer 1
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer1_eq_hid_inc(LAYER_HIDDEN1_SIZE, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer1_eq_hid_dec(LAYER_HIDDEN1_SIZE, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NNB_LayersAggregator layer1_eq_hid({ &layer1_eq_hid_inc, &layer1_eq_hid_dec });
	// Heading summator
	nn::NeuronHoldingStaticLayer<nnLinSumm> layer15_eq_hid_head(LAYER_HIDDEN1_SIZE, [&](nnLinSumm *const mem_ptr, unsigned) {
		new(mem_ptr)nnLinSumm;
	});

	// Equivalent Hidden layer 2
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer2_eq_hid_inc(LAYER_HIDDEN2_SIZE, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer2_eq_hid_dec(LAYER_HIDDEN2_SIZE, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NNB_LayersAggregator layer2_eq_hid({ &layer2_eq_hid_inc, &layer2_eq_hid_dec });
	// Heading summator
	nn::NeuronHoldingStaticLayer<nnLinSumm> layer25_eq_hid_head(LAYER_HIDDEN2_SIZE, [&](nnLinSumm *const mem_ptr, unsigned) {
		new(mem_ptr)nnLinSumm;
	});


	// Equivalent Output layer
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer3_eq_out_inc(10, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer3_eq_out_dec(10, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NNB_LayersAggregator layer3_eq_out({ &layer3_eq_out_inc, &layer3_eq_out_dec });
	// Heading summator
	nn::NeuronHoldingStaticLayer<nnLinSumm> layer35_eq_out_head(10, [&](nnLinSumm *const mem_ptr, unsigned) {
		new(mem_ptr)nnLinSumm;
	});

	using OptimAlg = nn::optimizers::Adam;
	OptimAlg optimizer(LEARNING_RATE);

	// Connections of Non-Monotinic (weights initialization is not required if equivalent trained first)
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmhid1(&layer_bias, &layer1_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmhid2(&layer_bias, &layer2_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmout(&layer_bias, &layer3_nm_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	// Non-Monotinic FeedForward connections
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_nmhid1(&layer_inp, &layer1_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_nmhid1_nmhid2(&layer1_nm_hid, &layer2_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_nmhid2_nmout(&layer2_nm_hid, &layer3_nm_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	// Connections of Equivalent
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_eqhid1(&layer_bias, &layer1_eq_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_eqhid2(&layer_bias, &layer2_eq_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_eqout(&layer_bias, &layer3_eq_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor_diviation(preudorandom));
	});
	// Equivalent FeedForward connections
	// Input -> Hidden-1 EQ1
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_eqhid1_inc(&layer_inp, &layer1_eq_hid_inc, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	// Input -> Hidden-1 EQ2
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_eqhid1_dec(&layer_inp, &layer1_eq_hid_dec, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -randistributor(preudorandom));
	});
	// (Hidden-1 EQ1 + Hidden-1 EQ2) -> Hidden-1 Head
	nn::SparceLayerStaticConnectomHolder2Mult<nn::NNB_StraightConnection> connections_inp_eqhid1_head(&layer1_eq_hid, &layer15_eq_hid_head, [&](nn::NNB_StraightConnection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_StraightConnection(from, to);
	});
	// Hidden-1 Head -> Hidden-2 EQ1
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_eqhid1_eqhid2_inc(&layer15_eq_hid_head, &layer2_eq_hid_inc, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	// Hidden-1 Head -> Hidden-2 EQ2
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_eqhid1_eqhid2_dec(&layer15_eq_hid_head, &layer2_eq_hid_dec, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -randistributor(preudorandom));
	});
	// (Hidden-2 EQ1 + Hidden-2 EQ2) -> Hidden-2 Head
	nn::SparceLayerStaticConnectomHolder2Mult<nn::NNB_StraightConnection> connections_eqhid1_eqhid2_head(&layer2_eq_hid, &layer25_eq_hid_head, [&](nn::NNB_StraightConnection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_StraightConnection(from, to);
	});

	// Hidden-2 Head -> Out EQ1
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_eqhid2_out_inc(&layer25_eq_hid_head, &layer3_eq_out_inc, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	// Hidden-2 Head -> Out EQ2
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_eqhid2_out_dec(&layer25_eq_hid_head, &layer3_eq_out_dec, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -randistributor(preudorandom));
	});
	// (Out EQ1 + Out EQ2) -> Out Head
	nn::SparceLayerStaticConnectomHolder2Mult<nn::NNB_StraightConnection> connection_eqhid2_out_head(&layer3_eq_out, &layer35_eq_out_head, [&](nn::NNB_StraightConnection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_StraightConnection(from, to);
	});

	// Projecting accessory
	nn::Monotonic2FieldsHeuristicsEqExV2 mon2filds(PROJECTION_THRESHOLD);
	nn::Monotonic2FieldsProjectingAccessory projector_lr1(&layer1_nm_hid, &layer1_eq_hid, &mon2filds);
	nn::Monotonic2FieldsProjectingAccessory projector_lr2(&layer2_nm_hid, &layer2_eq_hid, &mon2filds);
	nn::Monotonic2FieldsProjectingAccessory projector_lr3(&layer3_nm_out, &layer3_eq_out, &mon2filds);


	nn::LearnGuiderFwBPgThreadAble learnguider_mtr_nm({ &layer1_nm_hid, &layer2_nm_hid, &layer3_nm_out }, BATCH_SIZE, THREADS_COUNT); // Evaluator & backpropagator for training
	nn::LearnGuiderFwBPgThreadAble learnguider_mtr_equiv({ &layer1_eq_hid, &layer15_eq_hid_head, &layer2_eq_hid, &layer25_eq_hid_head, &layer3_eq_out, &layer35_eq_out_head }, BATCH_SIZE, THREADS_COUNT); // Evaluator & backpropagator for training

	nn::LearnGuiderFwBPg learnguider_sing_nm({ &layer1_nm_hid, &layer2_nm_hid, &layer3_nm_out }); // Evaluator for validating & testing
	nn::LearnGuiderFwBPg learnguider_sing_equiv({ &layer1_eq_hid, &layer15_eq_hid_head, &layer2_eq_hid, &layer25_eq_hid_head, &layer3_eq_out, &layer35_eq_out_head }); // Evaluator for validating & testing

	const float UINT8_NORM = 1.0f / 255.0f;

	// Learning
	std::vector<std::vector<float>> perfect_out_store(BATCH_SIZE, std::vector<float>(10));
	std::vector<unsigned short> perfect_outs(BATCH_SIZE);

	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_train_calculator(mnist_train_classes_w.GetWeights());
	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator(mnist_test_classes_w.GetWeights());

	std::cout << std::fixed << std::setprecision(3);

	std::atomic_flag data_is_ready_for_workers;
	std::atomic_uint workers_is_ready_for_data;
	bool threads_run = true;

	bool learn_nonmonotonic = false;
	bool calc_loss = false;
	std::atomic<float> threads_loss;

	auto TrainWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_mtr_nm.GetRequiredCachesCount());
		nn::errcalc::ErrorCalcSoftMAX softmax_calculator(10);

		float loss;
		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			loss = 0.0f;
			// Learning
			if (learn_nonmonotonic) {
				learnguider_mtr_nm.WorkerDoForward(worker_id, &caches[0]);
				loss += learnguider_mtr_nm.FillupOutsError(worker_id, &softmax_calculator, perfect_out_store, calc_loss);
				learnguider_mtr_nm.WorkerDoBackward(worker_id, &caches[0]);
			} else {
				learnguider_mtr_equiv.WorkerDoForward(worker_id, &caches[0]);
				loss += learnguider_mtr_equiv.FillupOutsError(worker_id, &softmax_calculator, perfect_out_store, calc_loss);
				learnguider_mtr_equiv.WorkerDoBackward(worker_id, &caches[0]);
			}

			// Loss & local accuracy
			if (calc_loss) {
				f1score_train_calculator.AppendResultsThreadSafe((learn_nonmonotonic ? learnguider_mtr_nm.GetOutputs() : learnguider_mtr_equiv.GetOutputs()), perfect_out_store, worker_id, THREADS_COUNT);
				threads_loss.fetch_add(loss, std::memory_order_relaxed);
			}

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	std::vector<std::thread> workers;
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TrainWorkerThread, i);
	}

	unsigned current_threads_count;
	unsigned log_iterations = 0;
	unsigned batch_iterations = 0;
	unsigned batch_iterations_all = mnist_train.size() / BATCH_SIZE;
	for (unsigned epoch = 0; epoch != EPOCHS; ++epoch) {
		std::shuffle(mnist_train.begin(), mnist_train.end(), preudorandom);

		auto datase_current_pos = mnist_train.cbegin();
		auto datase_end = mnist_train.cend();

		batch_iterations = 0;
		// Training epoch
		do {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				auto &out = perfect_out_store[batch_i];
				std::fill(out.begin(), out.end(), 0.0f);
				out[datase_current_pos->label] = 1.0f;
			}
			++batch_iterations;

			calc_loss = false;
			if (++log_iterations == LOG_INTERVAL) {
				log_iterations = 0;
				calc_loss = true;
			}

			data_is_ready_for_workers.test_and_set(std::memory_order_release);
			data_is_ready_for_workers.notify_all();

			// Training is performing now

			current_threads_count = 0;
			do {
				workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
				current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
			} while (current_threads_count != THREADS_COUNT);
			workers_is_ready_for_data.store(0, std::memory_order_release);

			if (calc_loss) {
				float loss = threads_loss.load(std::memory_order_relaxed);
				threads_loss.store(0.0f, std::memory_order_relaxed);
				loss /= BATCH_SIZE;
				std::cout << "Epoch: " << (epoch + 1) << "  Batch: " << std::setfill('0') << std::setw(3) << batch_iterations << '/' << batch_iterations_all << "  Loss: " << loss << "  Accuracy(local): " << f1score_train_calculator.CalcAccuracy() << "%  F1(local): " << f1score_train_calculator.CalcF1() << "\n";
				f1score_train_calculator.Reset();
			}
		} while (datase_current_pos != datase_end);

		// Fast validation after epoch
		datase_current_pos = mnist_valid.cbegin();
		datase_end = mnist_valid.cend();
		while (true) {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				perfect_outs[batch_i] = datase_current_pos->label;
			}
			if (datase_current_pos == datase_end) {
				break;
			}

			if (learn_nonmonotonic) {
				learnguider_sing_nm.DoForward();
				for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
					f1score_test_calculator.AppendResult(nn::netquality::NeuroArgmax(learnguider_sing_nm.GetOutputs(), batch_i), perfect_outs[batch_i]);
				}
			} else {
				learnguider_sing_equiv.DoForward();
				for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
					f1score_test_calculator.AppendResult(nn::netquality::NeuroArgmax(learnguider_sing_equiv.GetOutputs(), batch_i), perfect_outs[batch_i]);
				}
			}
		}
		std::cout << (learn_nonmonotonic ? "<NON-MONOTONIC>" : "<EQUIVALENT>") << "\t\t\t\tAccuracy (validation dataset): " << f1score_test_calculator.CalcAccuracy() << "%  F1: " << f1score_test_calculator.CalcF1() << "\n";
		f1score_test_calculator.Reset();

		if (epoch == 1) {
			std::cout << "Performing EQUIVALENT => NON-MONOTONIC projecting...\n";
			projector_lr1.Perform2to1LossyCompression();
			projector_lr2.Perform2to1LossyCompression();
			projector_lr3.Perform2to1LossyCompression();
			learn_nonmonotonic = true;
		}
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();
	// Ensuring all released
	for (auto &thr : workers) {
		thr.join();
	}

	// Non-monotonity usage stats: minus_count, plus_count, min, summ, max
	std::map<const nnNmReLU *, std::tuple<unsigned, unsigned, float, float, float>> fu_both_neurons;

	auto NonMonotonityStatProc = [&](nnNmReLU *nrn) {
		auto iter = fu_both_neurons.find(nrn);
		if (iter != fu_both_neurons.end()) {
			for (unsigned batch_i = 0; batch_i != nrn->GetCurrentBatchSize(); ++batch_i) {
				if (nrn->RealAccumulatorValue(batch_i) < 0) {
					++std::get<0>(iter->second);
				} else {
					++std::get<1>(iter->second);
				}
				std::get<3>(iter->second) += nrn->RealAccumulatorValue(batch_i);
				if (nrn->RealAccumulatorValue(batch_i) < std::get<2>(iter->second)) {
					std::get<2>(iter->second) = nrn->RealAccumulatorValue(batch_i);
				} else if (nrn->RealAccumulatorValue(batch_i) > std::get<4>(iter->second)) {
					std::get<4>(iter->second) = nrn->RealAccumulatorValue(batch_i);
				}
			}
		} else {
			{
				bool minus = nrn->RealAccumulatorValue() < 0;
				fu_both_neurons.emplace(nrn, std::make_tuple((unsigned)(minus), (unsigned)(!minus), nrn->RealAccumulatorValue(), nrn->RealAccumulatorValue(), nrn->RealAccumulatorValue()));
			}
			iter = fu_both_neurons.find(nrn);
			for (unsigned batch_i = 1; batch_i < nrn->GetCurrentBatchSize(); ++batch_i) {
				if (nrn->RealAccumulatorValue(batch_i) < 0) {
					++std::get<0>(iter->second);
				} else {
					++std::get<1>(iter->second);
				}
				std::get<3>(iter->second) += nrn->RealAccumulatorValue(batch_i);
				if (nrn->RealAccumulatorValue(batch_i) < std::get<2>(iter->second)) {
					std::get<2>(iter->second) = nrn->RealAccumulatorValue(batch_i);
				} else if (nrn->RealAccumulatorValue(batch_i) > std::get<4>(iter->second)) {
					std::get<4>(iter->second) = nrn->RealAccumulatorValue(batch_i);
				}
			}
		}
	};


	std::cout << "Accuracy calculating...\n";

	// Accuracy calc
	unsigned half_THREADS_COUNT = THREADS_COUNT >> 1;
	unsigned full_THREADS_COUNT = half_THREADS_COUNT << 1;
	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator_nm(mnist_test_classes_w.GetWeights());

	learnguider_mtr_nm.SetThreadsCount(half_THREADS_COUNT);
	learnguider_mtr_equiv.SetThreadsCount(half_THREADS_COUNT);
	
	auto TestWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_mtr_nm.GetRequiredCachesCount(false));
		std::vector<float> *caches_ptr = caches.size() ? &caches[0] : nullptr;

		unsigned sub_id = worker_id >> 1;
		bool nonmonotonic = worker_id & 1;

		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			if (nonmonotonic) {
				learnguider_mtr_nm.WorkerDoForward(sub_id, caches_ptr);
				f1score_test_calculator_nm.AppendResultsThreadSafe(learnguider_mtr_nm.GetOutputs(), perfect_outs, sub_id, half_THREADS_COUNT);
			} else {
				learnguider_mtr_equiv.WorkerDoForward(sub_id, caches_ptr);
				f1score_test_calculator.AppendResultsThreadSafe(learnguider_mtr_equiv.GetOutputs(), perfect_outs, sub_id, half_THREADS_COUNT);
			}

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == full_THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	// Threads prepare
	data_is_ready_for_workers.clear();
	threads_run = true;

	workers.clear();
	for (unsigned i = 0; i != full_THREADS_COUNT; ++i) {
		workers.emplace_back(TestWorkerThread, i);
	}

	// Accuracy calc
	unsigned batch_count = 0;
	auto test_current_pos = mnist_test.cbegin();
	auto test_end = mnist_test.cend();
	batch_iterations_all = mnist_test.size() / BATCH_SIZE;

	log_iterations = 0;
	while (true) {
		// Update inputs
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++test_current_pos) {
			if (test_current_pos == test_end) {
				break;
			}
			auto iter = inputs_store[batch_i].begin();
			for (auto item : test_current_pos->input) {
				*iter = UINT8_NORM * item;
				++iter;
			}
			perfect_outs[batch_i] = test_current_pos->label;
		}
		if (test_current_pos == test_end) {
			break;
		}
		++batch_count;
		if (++log_iterations == LOG_INTERVAL) {
			log_iterations = 0;
			std::cout << "Batch: " << batch_count << '/' << batch_iterations_all << '\n';
		}

		data_is_ready_for_workers.test_and_set(std::memory_order_release);
		data_is_ready_for_workers.notify_all();

		// Evaluating is performing now

		current_threads_count = 0;
		do {
			workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
			current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
		} while (current_threads_count != full_THREADS_COUNT);
		workers_is_ready_for_data.store(0, std::memory_order_release);

		// Grab non-monotonity stat
		for (auto lr : learnguider_mtr_nm.GetLayers()) {
			for (auto nrn : lr->Neurons()) {
				NonMonotonityStatProc(dynamic_cast<nnNmReLU *>(nrn));
			}
		}
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();

	std::cout << "Accuracy  <EQUIVALENT>   (test dataset): " << f1score_test_calculator.CalcAccuracy() << "%  F1: " << f1score_test_calculator.CalcF1() << "\n";
	std::cout << "Accuracy <NON-MONOTONIC> (test dataset): " << f1score_test_calculator_nm.CalcAccuracy() << "%  F1: " << f1score_test_calculator_nm.CalcF1() << "\n";

	std::cout << "F1-score per class:\n";
	std::cout << "Class\tEQUIV\tNON-M\n";
	for (short i = 0; i != 10; ++i) {
		std::cout << i << '\t' << f1score_test_calculator.CalcF1ForClass(i) << '\t' << f1score_test_calculator_nm.CalcF1ForClass(i) << '\n';
	}

	for (auto &thr : workers) {
		thr.join();
	}

	std::cout << "\nValues distribution (across non-monotonic neurons, X normalized by NM1_MAX):\n";
	
	float min_prc = 200.0f, max_prc = -1.0f, avg_prc = 0.0f;
	float min_X = std::numeric_limits<float>::max(), max_X = std::numeric_limits<float>::min(), avg_X = 0.0f;
	for (auto &iter : fu_both_neurons) {
		unsigned overall = std::get<0>(iter.second) + std::get<1>(iter.second);
		std::get<3>(iter.second) /= overall * NM1_MAX_VALUE; // Normalizing average
		std::get<2>(iter.second) /= NM1_MAX_VALUE; // Normalizing miniumm
		std::get<4>(iter.second) /= NM1_MAX_VALUE; // Normalizing maximum
		float balance = 100.f * std::get<1>(iter.second) / overall;

		avg_prc += balance;
		if (balance < min_prc) min_prc = balance;
		else if (balance > max_prc) max_prc = balance;

		avg_X += std::get<3>(iter.second);
		if (std::get<2>(iter.second) < min_X)  min_X = std::get<2>(iter.second);
		if (std::get<4>(iter.second) > max_X)  max_X = std::get<4>(iter.second);
	}
	avg_X /= fu_both_neurons.size();
	avg_prc /= fu_both_neurons.size();
	std::cout << "\tMin %\tMax %\tAvg %\tMin X\tAvg X\tMax X\n";
	std::cout << "Summ:\t" << min_prc << '\t' << max_prc << '\t' << avg_prc << '\t' << min_X << '\t' << avg_X << '\t' << max_X << '\n';

	std::cout << "Nrn ID\tNeg #\tPos #\tPos %\tMin X\tAvg X\tMax X\n";
	unsigned nrn = 0;
	for (auto &iter : fu_both_neurons) {
		float balance = 100.f * std::get<1>(iter.second) / (std::get<0>(iter.second) + std::get<1>(iter.second));
		std::cout << ++nrn << '\t' << std::get<0>(iter.second) << '\t' << std::get<1>(iter.second) << '\t' << balance << '\t' << std::get<2>(iter.second) << '\t' << std::get<3>(iter.second) << '\t' << std::get<4>(iter.second) << '\n';
	}
	return;
}