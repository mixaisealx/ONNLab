#include "onnlab.h"

#include <iostream>

#include "NNB_Connection_spyable.h"
#include "NNB_ReLU.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"

#include "ReverseGuiderB1.h"

#include <vector>


void exp_ReLU_revB1() {
	std::cout << "exp_ReLU_revB1" << std::endl;

	std::vector<float> inputs = { 0, 0 };
	nn::NNB_Input in1(&inputs[0]), in2(&inputs[1]);
	nn::NNB_ReLU lr11, lr12;
	nn::NNB_ReLU out;

	nn::NNB_Connection connections[] = {
		nn::NNB_Connection(&in1, &lr11),
		nn::NNB_Connection(&in1, &lr12),
		nn::NNB_Connection(&in2, &lr11),
		nn::NNB_Connection(&in2, &lr12),
		nn::NNB_Connection(&lr11, &out),
		nn::NNB_Connection(&lr12, &out)
	};

	nn::NNB_Layer layer1({ &lr11, &lr12 });
	nn::NNB_Layer layer2({ &out });

	connections[0].Weight(-0.932523727f);
	connections[1].Weight(0.922165334f);
	connections[2].Weight(0.937476695f);
	connections[3].Weight(-0.971143901f);
	connections[4].Weight(1.19150949f);
	connections[5].Weight(1.20489347f);

	nn::reverse::ReverseGuiderB1 revg({ &layer1, &layer2 });
	revg.FillTargetExactOutputs({ 1 });
	revg.ApplyLayerSolver(&layer1, &layer2);

	return;
}