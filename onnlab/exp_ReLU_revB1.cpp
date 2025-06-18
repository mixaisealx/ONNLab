#include "onnlab.h"

#include "NNB_Connection_spyable.h"
#include "OptimizerGD.h"
#include "NNB_ReLU.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"

#include "ReverseB1.h"

#include <vector>

#include <iostream>


static float UnReLU(float y, float alpha) {
	return (y < 0 ? y / alpha : y);
}

void exp_ReLU_revB1() {
	std::cout << "exp_ReLU_revB1" << std::endl;

	std::vector<float> inputs = { 0, 0 };
	nn::NNB_Input in1(&inputs[0]), in2(&inputs[1]);
	nn::NNB_ReLU lr11(0.1f), lr12(0.1f);
	nn::NNB_ReLU out(0.1f);

	using OptimGD = nn::optimizers::GradientDescendent;
	OptimGD optimizerGD;

	nn::NNB_Connection<OptimGD> connections[] = {
		nn::NNB_Connection<OptimGD>(&in1, &lr11, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&in1, &lr12, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&in2, &lr11, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&in2, &lr12, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&lr11, &out, &optimizerGD),
		nn::NNB_Connection<OptimGD>(&lr12, &out, &optimizerGD)
	};

	nn::NNB_Layer layer1({ &lr11, &lr12 });
	nn::NNB_Layer layer2({ &out });

	connections[0].Weight(-0.932523727f);
	connections[1].Weight(0.922165334f);
	connections[2].Weight(0.937476695f);
	connections[3].Weight(-0.971143901f);
	connections[4].Weight(1.19150949f);
	connections[5].Weight(1.20489347f);

	nn::reverse::ReverseB1 revb;
	revb.SetLayerReverseActivationFunction(&layer2, std::bind(UnReLU, std::placeholders::_1, 0.1f));
	revb.FillTargetExactOutputs({ 1 }, &layer2);
	revb.ApplyLayerSolver(&layer1, &layer2);

	return;
}