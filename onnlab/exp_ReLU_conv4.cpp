#include "onnlab.h"

#include "NNB_Connection.h"
#include "HyperConnection.h"
#include "OptimizerAdam.h"
#include "OptimizerGD.h"
#include "NNB_ReLU.h"
#include "NNB_Input.h"
#include "NNB_Storage.h"
#include "NNB_ConstInput.h"
#include "NNB_Layer.h"
#include "NNB_ConvolutionHead.h"
#include "NNB_ConvolutionEssence2d.h"
#include "LearnGuiderFwBPgThreadAble.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "NeuronHoldingStaticLayer.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <thread>

#include <iostream>


void exp_ReLU_conv4() {
	std::cout << "exp_ReLU_conv4" << std::endl;

	const unsigned BATCH_SIZE = 6;
	const unsigned THREADS_COUNT = 32;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);

	// Input vector
	std::array<std::array<float, 16>, BATCH_SIZE> inputs_store;

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(16, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});

	// Input from convolution
	nn::NeuronHoldingStaticLayer<nn::NNB_StorageB<BATCH_SIZE>> layer_inp2(9, [&](nn::NNB_StorageB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_StorageB<BATCH_SIZE>;
	});

	//Hidden
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> layer_hidden(16, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});

	// Output
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> layer_out(2, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});

	const float LEARNING_RATE = 0.1f;

	using OptimAlg = nn::optimizers::Adam;
	OptimAlg optimizer(LEARNING_RATE);

	// Connections
	// layer_inp -> layer_inp2
	nn::NNB_ConvolutionEssence2d conv2d_frame(&layer_inp, &layer_inp2, 4,
													nn::NNB_ConvolutionEssence2d::KernelShape(2), nn::NNB_ConvolutionEssence2d::KernelMovement(), 1);
	nn::NNB_ConvolutionHead<OptimAlg> layer_conv2d(&conv2d_frame, &optimizer, [&](bool) { return randistributor(preudorandom); }, false);

	// Hidden
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_inp2_hid(&layer_inp2, &layer_hidden, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	// To out
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_hid_out(&layer_hidden, &layer_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	std::uniform_int_distribution<int> noisegen(15, 240);
	std::uniform_int_distribution<unsigned> posegen(0, 2);

	// Test generator
	auto GenTest = [&](bool right, unsigned idx) {
		auto &input = inputs_store[idx];
		// Note: explicit to float/int cast due to suppress annoying warnings
		for (auto &elm : input) {
			elm = (float)noisegen(preudorandom);
		}

		if (right) {
			int posx = posegen(preudorandom);
			int posy = posegen(preudorandom);
			input[(posy) * 4 + posx] = input[(posy+1) * 4 + posx+1] = 0;
			input[(posy+1) * 4 + posx] = input[(posy) * 4 + posx + 1] = 255;
		}

		for (auto &elm : input) {
			elm /= 255;
		}
	};

	// Input to
	nn::errcalc::ErrorCalcSoftMAX softmax_calculator(2);
	nn::LearnGuiderFwBPgThreadAble learnguider_mthr({ &layer_conv2d, &layer_hidden, &layer_out }, BATCH_SIZE, THREADS_COUNT);
	nn::LearnGuiderFwBPg learnguider({ &layer_conv2d, &layer_hidden, &layer_out }, &softmax_calculator, BATCH_SIZE);

	std::uniform_int_distribution<unsigned> testselector(0, 1);

	std::vector<std::vector<float>> perfect_out_store(BATCH_SIZE, std::vector<float>(2));

	std::atomic_flag data_is_ready_for_workers;
	std::atomic_uint workers_is_ready_for_data;
	bool threads_run = true;

	auto TrainWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_mthr.GetRequiredCachesCount());
		nn::errcalc::ErrorCalcSoftMAX softmax_calculator(2);

		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			// Learning
			learnguider_mthr.WorkerDoForward(worker_id, &caches[0]);
			learnguider_mthr.FillupOutsError(worker_id, &softmax_calculator, perfect_out_store);
			learnguider_mthr.WorkerDoBackward(worker_id, &caches[0]);

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

	size_t OVERALL_ITERATIONS = 15000;
	for (size_t iterations = 0, iters = OVERALL_ITERATIONS / BATCH_SIZE; iterations < iters; ++iterations) {
		// Update inputs
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			bool right = testselector(preudorandom) > 0;
			perfect_out_store[batch_i][0] = !right;
			perfect_out_store[batch_i][1] = right;

			GenTest(right, batch_i);
		}

		data_is_ready_for_workers.test_and_set(std::memory_order_release);
		data_is_ready_for_workers.notify_all();

		// Training is performing now

		unsigned current_threads_count = 0;
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

	// Accuracy calc
	size_t iters = 1000;
	int rcount = 0;
	for (size_t iterations = 0; iterations < iters; ++iterations) {
		// Select datarow
		bool right = testselector(preudorandom) == 0;

		// Update inputs
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			GenTest(right, batch_i);
		}

		learnguider.DoForward();
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			rcount += right == (learnguider.GetOutputs()[0]->OwnLevel(batch_i) < learnguider.GetOutputs()[1]->OwnLevel(batch_i));
		}
	}

	std::cout << "Accuracy: " << rcount / (float)(iters * BATCH_SIZE) * 100 << "%\n";

	for (auto &thr : workers) {
		thr.join();
	}
	return;
}