void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc){
	/*signalPower = get_signal_power(ADCData2, ADC_BUF_SIZE);
	printf("%d", signalPower);
	printf("\n");
	if(lastSample == 1){
		HAL_TIM_Base_Stop(&htim3);
		HAL_ADC_Stop_DMA(&hadc1);
		lastSample = 0;
		state = 0;
		print_buffer(ADCData1);
	}
	if(signalPower > 50000){
		lastSample = 1;
	}*/
	HAL_TIM_Base_Stop(&htim3);
	HAL_ADC_Stop_DMA(&hadc1);
	printf("Stop listening, sending the buffer.\n");
	print_buffer(ADCData1);
	//print_buffer(ADCData2);
}
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
	if (GPIO_Pin == B1_Pin) {
		state = 1-state;
	}
	if (state == 1){
		printf("Is listening.\n");
		HAL_TIM_Base_Start(&htim3);
		HAL_ADC_Start_DMA(&hadc1, ADCData1, ADC_BUF_SIZE*2);
		state = 0;
	}
}