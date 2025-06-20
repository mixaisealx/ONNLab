#include "onnlab.h"

#include "NNB_Connection.h"
#include "OptimizerGD.h"
#include "OptimizerAdam.h"
#include "NNB_Linear.h"
#include "NNB_iReLU.h"
#include "NNB_Input.h"
#include "NNB_ConstInput.h"
#include "NNB_Layer.h"
#include "NNB_LinearSlim.h"
#include "LearnGuiderFwBPgThreadAble.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "NeuronHoldingStaticLayer.h"

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


void exp_iReLU_mnist1() {
	std::cout << "exp_iReLU_mnist1" << std::endl;

	const unsigned THREADS_COUNT = 32;
	const unsigned LOG_INTERVAL = 30;

	const unsigned BATCH_SIZE = 128;
	const unsigned EPOCHS = 5;
	const float VALID_TEST_SPLIT_RATIO = 0.1f; // sizeof(validation)/sizeof(full testing dataset size), should be small to run fast

	const float LEARNING_RATE = 0.01f;

	const unsigned LAYER_HIDDEN_SIZE = 8;

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
	std::mt19937 preudorandom(42);

	std::shuffle(mnist_test.begin(), mnist_test.end(), preudorandom);
	std::vector<MNIST_entry> mnist_valid;
	mnist_valid.reserve(static_cast<unsigned>(VALID_TEST_SPLIT_RATIO * mnist_test.size()));
	{
		std::copy_n(mnist_test.begin(), mnist_valid.capacity(), std::back_inserter(mnist_valid));
	}

	std::uniform_real_distribution<float> randistributor(0.01f, 0.1f);

	const unsigned INPUT_SIZE = 28 * 28;

	using nnReLU = nn::NNB_iReLUb<BATCH_SIZE, true>; // Using Kahan summation

	// Input image (like vector, flatten)
	std::vector<std::array<float, INPUT_SIZE>> inputs_store(BATCH_SIZE);

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(INPUT_SIZE, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});

	// Hidden layer
	nn::NeuronHoldingStaticLayer<nnReLU> layer_hid(LAYER_HIDDEN_SIZE, [&](nnReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnReLU;
	});

	// Output layer
	nn::NeuronHoldingStaticLayer<nnReLU> layer_out(10, [&](nnReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnReLU;
	});

	using OptimAlg = nn::optimizers::Adam;
	OptimAlg optimizer(LEARNING_RATE);
	
	// Connections
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_bias_hid(&layer_bias, &layer_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_bias_out(&layer_bias, &layer_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	// Input to
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_hid(&layer_inp, &layer_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_hid_out(&layer_hid, &layer_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	nn::LearnGuiderFwBPgThreadAble learnguider_multhr({ &layer_hid, &layer_out }, BATCH_SIZE, THREADS_COUNT); // Evaluator & backpropagator for training
	nn::LearnGuiderFwBPg learnguider_single({ &layer_hid, &layer_out }, BATCH_SIZE); // Evaluator for validating & testing

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

	bool calc_loss = false;
	std::atomic<float> threads_loss;
	
	auto TrainWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_multhr.GetRequiredCachesCount());
		nn::errcalc::ErrorCalcSoftMAX softmax_calculator(10);

		float loss;
		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run) 
				break;
			
			loss = 0.0f;
			// Learning
			learnguider_multhr.WorkerDoForward(worker_id, &caches[0]);
			loss += learnguider_multhr.WorkerFillupOutsError(worker_id, &softmax_calculator, perfect_out_store, calc_loss);
			learnguider_multhr.WorkerDoBackward(worker_id, &caches[0]);
			
			// Loss & local accuracy
			if (calc_loss) {
				f1score_train_calculator.AppendResultsThreadSafe(learnguider_multhr.GetOutputs(), perfect_out_store, worker_id, THREADS_COUNT);
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
				float f1 = f1score_train_calculator.CalcF1();
				float accuracy = f1score_train_calculator.CalcAccuracy();
				f1score_train_calculator.Reset();
				loss /= BATCH_SIZE;
				std::cout << "Epoch: " << (epoch + 1) << "  Batch: " << std::setfill('0') << std::setw(3) << batch_iterations << '/' << batch_iterations_all << "  Loss: " << loss << "  Accuracy(local): " << accuracy << "%  F1(local): " << f1 << "\n";
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

			learnguider_single.DoForward();

			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
				f1score_test_calculator.AppendResult(nn::netquality::NeuroArgmax(learnguider_single.GetOutputs(), batch_i), perfect_outs[batch_i]);
			}
		}
		std::cout << "\t\t\t\t\tAccuracy (validation dataset): " << f1score_test_calculator.CalcAccuracy() << "%  F1: " << f1score_test_calculator.CalcF1() << "\n";
		f1score_test_calculator.Reset();
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();
	// Ensuring all released
	for (auto &thr : workers) {
		thr.join();
	}
	

	std::cout << "Accuracy calculating...\n";

	// Accuracy calc
	auto TestWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_multhr.GetRequiredCachesCount(false));
		std::vector<float> *caches_ptr = caches.size() ? &caches[0] : nullptr;

		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			learnguider_multhr.WorkerDoForward(worker_id, caches_ptr);
			f1score_test_calculator.AppendResultsThreadSafe(learnguider_multhr.GetOutputs(), perfect_outs, worker_id, THREADS_COUNT);

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	// Threads prepare
	data_is_ready_for_workers.clear();
	threads_run = true;

	workers.clear();
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TestWorkerThread, i);
	}

	unsigned batch_count = 0;
	auto test_current_pos = mnist_test.cbegin();
	auto test_end = mnist_test.cend();
	batch_iterations_all = mnist_test.size() / BATCH_SIZE;
	
	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator000(mnist_test_classes_w.GetWeights());

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
		} while (current_threads_count != THREADS_COUNT);
		workers_is_ready_for_data.store(0, std::memory_order_release);
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();

	std::cout << "Accuracy (test dataset): " << f1score_test_calculator.CalcAccuracy() << "%  F1: " << f1score_test_calculator.CalcF1() << "\n";
	std::cout << "F1-score per class:\n";
	for (short i = 0; i != 10; ++i) {
		std::cout << "Class " << i << ": " << f1score_test_calculator.CalcF1ForClass(i) << '\n';
	}

	f1score_test_calculator.Reset();

	for (auto &thr : workers) {
		thr.join();
	}
	return;
}