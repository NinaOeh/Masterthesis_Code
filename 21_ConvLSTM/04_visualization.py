#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd

#%%


# Visualizing the val_loss convergence of the two models for all three optimization rounds
### LSTM MODEL ###

LSTM_init_opt_loss_ = [0.11534601703286171, 0.15100119829177858, 0.1481168645620346, 0.15100119829177858, 0.1481168645620346, 0.13530903398990632, 0.10545781522989273, 0.12907760381698608,
0.1202102243900299, 0.11732859939336776, 0.11806652545928956, 0.14362720489501954, 0.18014955401420593, 0.10689112305641174, 0.11641745656728744, 0.1116730760037899, 0.17525187849998475, 0.13447513222694396, 0.10842568099498749, 0.13976158142089845, 0.12348813593387603, 0.1212417510151863, 0.325687849521637, 0.1576099395751953, 0.11275146961212158, 0.1552150970697403, 0.11658831179141999, 1.1850651025772094, 0.17044214010238648, 0.11412149727344513, 0.1109993976354599, 0.11637912392616272, 0.12101535737514496, 0.11029252618551254, 0.13862656444311142, 0.11188330292701722, 0.10752026140689849, 0.1200150865316391, 0.1314651796221733, 0.1066814860701561]
LSTM_log_refine_loss_ = [0.11870896875858307, 0.11581842541694641, 0.11468343496322632, 0.10220532774925233, 0.10789603233337403, 0.10666180014610291, 0.11719968259334564, 0.1576881742477417,  0.12420464485883713, 0.10733043611049652, 0.11194095134735108, 0.10786906719207763, 0.10595948040485383, 0.12548853754997252, 0.10720243692398071, 0.11116238415241242, 0.11506392121315002, 0.11378388166427612, 0.10768462359905243, 0.10166754990816117, 0.10497631013393402, 0.10427423506975174, 0.10367191284894943, 0.1047658383846283, 0.1132063889503479, 0.1073923310637474, 0.12666390627622603,  0.10906744092702865, 0.10566156536340714, 0.09996436417102814, 0.10700424879789353, 0.10952079862356186, 0.1045716518163681, 0.11197208046913147, 0.1282346147298813, 0.10535947889089585]
LSTM_normal_refine_loss_ = [0.16833260118961335, 0.12542222678661347, 0.11548763155937194, 0.1331148052215576, 
0.16756903290748595, 0.13997053861618042, 0.14975825279951097, 0.13950595676898955, 0.1302020889520645, 0.12311346173286437, 0.12178056806325913, 0.13112042874097823, 0.14510686576366424, 0.13119987428188323, 0.12264980733394623, 0.14065954685211182, 0.1281343385577202, 0.1303410002589226, 0.12324999511241913, 0.11007740914821625, 0.11897000551223755, 0.12439891457557678, 0.1086695882678032, 0.11212838470935821, 0.10743661671876907, 0.14907063245773317, 0.1304983013868332, 0.12584250628948213, 0.13052306830883026, 0.12785954594612123, 0.12981321692466735, 0.144487224817276,  0.1256083595752716, 0.12668040335178377,  0.1362404552102089, 0.14018715620040895, 0.15709981679916382]


LSTM_init_opt_loss = np.exp(np.array(LSTM_init_opt_loss_))
LSTM_log_refine_loss = np.exp(np.array(LSTM_log_refine_loss_))
LSTM_normal_refine_loss = np.exp(np.array(LSTM_normal_refine_loss_))

slope_11, intercept_11, r_value_11, p_value_11, std_err_11 = stats.linregress(range(len(LSTM_init_opt_loss)), LSTM_init_opt_loss)
regression_line_11 = slope_11 * np.array(range(len(LSTM_init_opt_loss))) + intercept_11

slope_12, intercept_12, r_value_12, p_value_12, std_err_12 = stats.linregress(range(len(LSTM_log_refine_loss)), LSTM_log_refine_loss)
regression_line_12 = slope_12 * np.array(range(len(LSTM_log_refine_loss))) + intercept_12

slope_13, intercept_13, r_value_13, p_value_13, std_err_13 = stats.linregress(range(len(LSTM_normal_refine_loss)), LSTM_normal_refine_loss)
regression_line_13 = slope_13 * np.array(range(len(LSTM_normal_refine_loss))) + intercept_13

plt.figure(figsize=(6, 3))
plt.plot(range(len(LSTM_init_opt_loss)), LSTM_init_opt_loss, label='initial optimization', color='red')
#plt.plot(range(len(LSTM_init_opt_loss)), regression_line_11, color='red', linestyle='--')
plt.plot(range(len(LSTM_log_refine_loss)),LSTM_log_refine_loss, label='logistic learning rate sampling refinement', color='blue')
#plt.plot(range(len(LSTM_log_refine_loss)), regression_line_12, color='blue', linestyle='--')
plt.plot(range(len(LSTM_normal_refine_loss)),LSTM_normal_refine_loss, label='normal learning rate sampling refinement', color='green')
#plt.plot(range(len(LSTM_normal_refine_loss)), regression_line_13, color='green', linestyle='--')
plt.xlabel('Optimization iteration')
plt.ylabel('Validation loss (MSE)')
plt.ylim(1.09,1.5)

min_val_loss_ = min(LSTM_log_refine_loss_)
min_lr = LSTM_log_refine_loss_.index(min_val_loss_)
min_val_loss = np.exp(min_val_loss_)
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)


plt.annotate(f"Minimal validation loss: \n{min_val_loss:.6f} ", 
             xy=(min_lr, min_val_loss), xytext=(min_lr, min_val_loss-0.11),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.title('ConvLSTM model \n validation loss development for model tuning')
plt.legend()
plt.grid()
plt.show()

#%%
### CONVSTM MODEL w. SA ###

LSTM_init_opt_loss_ = [0.0006964719586540014, 0.0008216162608005107, 0.0007146735093556345, 0.0006507922755554319, 0.0006726212229114025, 0.0007018773746676743, 
0.0006725508300587535, 0.0006656494410708547, 0.002026051958091557, 0.0008483249903656542, 0.004001061972230673, 0.0006543729058466852, 
0.0018023364758118986, 0.0006707748619373888, 0.0007953359698876739, 0.0007458931813016533, 0.0007106937724165618, 0.0006654819427058101, 0.0006791839306242764, 0.0007875422865618021, 0.000683556639123708, 0.0006867685494944453, 0.0007525262411218137, 0.0006627323594875634, 0.0006973577244207263, 0.0007166219665668905, 0.0007477700570598245, 0.0006430984230246394, 0.0007507112016901374, 0.0007077365834265947, 0.0008554863696917891, 0.0006537589826621116, 0.0007402214989997446, 0.000734008471481502, 0.0007566688547376543, 0.04991343513131142, 0.0007366381166502834, 0.0006470611644908786, 0.0006472721346653998, 0.000902340819593519, 0.0007221229438437149, 0.0008591192262247205, 0.000708099075127393]
LSTM_log_refine_loss_ = [0.0006932114739902317, 0.0018492840370163321, 0.0006886056368239224, 0.0006511089764535427, 0.0007447635265998543, 0.0007572694239206612, 
0.000664992043748498, 0.0006891482311766595, 0.0006642937287688255, 0.0006679086922667921, 0.0006606715789530427, 0.0007726194569841027, 0.0006763590482296423, 0.0006573969172313809, 0.0007005886500701308, 0.011824327036738396, 0.0007190803694538772, 0.0007198877539485693, 0.001840482451952994, 
0.0006536774057894945, 0.0007902339042630046, 0.0006640149070881307, 0.0007127697067335248, 0.0007099519483745098, 0.0007232943223789334, 0.000737323712091893, 0.0007324890862219035, 0.0006866031442768871, 0.0007812797836959362, 0.0006693201535381377, 0.0006889959145337343, 0.0006832486414350569, 0.0007282810937613249, 0.0006810337759088725, 0.0008009769534692168, 0.000642020518425852, 0.0006492644583340735, 0.0006270294287241996, 0.0006879671942442656, 0.0006694568623788655, 0.0006864911341108382, 0.0006908157607540488, 0.0006696952728088945]
LSTM_normal_refine_loss_ =  [0.001851619048975408, 0.001888351608067751, 0.002667052582837641, 0.0019421172211878002, 0.0018881483376026154, 0.0007720913086086511, 0.0007405377295799554, 0.00067883497744333, 0.0006968418043106794, 0.0019210136588662862, 0.0006634156068321317, 0.002064406406134367, 0.0006472776993177831, 0.0006589053652714938, 0.0006745514483191073, 0.0006581702921539545, 0.000651697147404775, 0.001878354293294251, 0.0006540565425530076, 0.0006686998088844121, 0.0022341293515637517, 0.0018550362065434456, 0.0006550406815949828, 0.001880033640190959, 0.0006679598160553723, 0.000686745154671371, 0.0007066698628477753, 0.0006464158929884434, 0.0006816101190634072, 0.0018529522931203246, 0.0010244838590733707, 0.000649299225769937, 0.0006713835150003434, 0.0006763767101801932, 0.0006609754730015993, 0.0007080187136307359, 0.0006462023290805519, 0.0006486583664081991, 0.0006506289704702794, 0.0006598811293952167]
LSTM_init_opt_loss = np.array(LSTM_init_opt_loss_)
LSTM_log_refine_loss = np.array(LSTM_log_refine_loss_)
LSTM_normal_refine_loss = np.array(LSTM_normal_refine_loss_)

chl_max = 171.9263916015625 
chl_min = 1.000003457069397
LSTM_init_opt_loss = LSTM_init_opt_loss * (chl_max - chl_min) + chl_min
LSTM_log_refine_loss = LSTM_log_refine_loss * (chl_max - chl_min) + chl_min
LSTM_normal_refine_loss = LSTM_normal_refine_loss * (chl_max - chl_min) + chl_min

slope_11, intercept_11, r_value_11, p_value_11, std_err_11 = stats.linregress(range(len(LSTM_init_opt_loss)), LSTM_init_opt_loss)
regression_line_11 = slope_11 * np.array(range(len(LSTM_init_opt_loss))) + intercept_11

slope_12, intercept_12, r_value_12, p_value_12, std_err_12 = stats.linregress(range(len(LSTM_log_refine_loss)), LSTM_log_refine_loss)
regression_line_12 = slope_12 * np.array(range(len(LSTM_log_refine_loss))) + intercept_12

slope_13, intercept_13, r_value_13, p_value_13, std_err_13 = stats.linregress(range(len(LSTM_normal_refine_loss)), LSTM_normal_refine_loss)
regression_line_13 = slope_13 * np.array(range(len(LSTM_normal_refine_loss))) + intercept_13

plt.figure(figsize=(6, 3))
plt.plot(range(len(LSTM_init_opt_loss)), LSTM_init_opt_loss, label='initial optimization', color='red')
plt.plot(range(len(LSTM_init_opt_loss)), regression_line_11, color='red', linestyle='--')
plt.plot(range(len(LSTM_log_refine_loss)),LSTM_log_refine_loss, label='logistic learning rate sampling refinement', color='blue')
plt.plot(range(len(LSTM_log_refine_loss)), regression_line_12, color='blue', linestyle='--')
plt.plot(range(len(LSTM_normal_refine_loss)),LSTM_normal_refine_loss, label='normal learning rate sampling refinement', color='green')
plt.plot(range(len(LSTM_normal_refine_loss)), regression_line_13, color='green', linestyle='--')
plt.xlabel('Optimization iteration')
plt.ylabel('Validation loss (MSE)')
plt.ylim(1.09,1.5)
min_val_loss_ = min(LSTM_log_refine_loss_)
min_lr = LSTM_log_refine_loss_.index(min_val_loss_)
min_val_loss = min_val_loss_* (chl_max - chl_min) + chl_min
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)


plt.annotate(f"Minimal validation loss: \n{min_val_loss:.6f} ", 
             xy=(min_lr, min_val_loss), xytext=(min_lr-6, min_val_loss-0.11),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.title('ConvLSTM w. SA model \n validation loss development for model tuning')
plt.legend()
plt.grid()
plt.show()

# %%
def cubic_func(x, b, c, d):
    return  b * x **2 + c * x + d 

popt_11, pcov_11 = curve_fit(cubic_func, np.array(range(len(LSTM_init_opt_loss))), LSTM_init_opt_loss)
regression_curve_11 = cubic_func(np.array(LSTM_init_opt_loss), *popt_11)

popt_12, pcov_12 = curve_fit(cubic_func, np.array(range(len(LSTM_log_refine_loss))), LSTM_log_refine_loss)
regression_curve_12 = cubic_func(np.array(LSTM_log_refine_loss), *popt_12)

plt.scatter(range(len(LSTM_init_opt_loss)), LSTM_init_opt_loss, label='Data Points')
plt.plot(range(len(LSTM_init_opt_loss)), regression_curve_11, color='red', label='Quadratic Regression')
plt.scatter(range(len(LSTM_log_refine_loss)), LSTM_log_refine_loss, label='Data Points')
plt.plot(range(len(LSTM_log_refine_loss)), regression_curve_12, color='red', label='Quadratic Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Quadratic Regression')
plt.legend()
plt.ylim(0.09,0.2)
plt.show()

# %%
import re

learning_rates = []
dropouts = []
filters = []
kernel_sizes_1 = []
kernel_sizes_2 = []
kernel_sizes_3 = []
activation_functions = []
kernel_size_finals = []
batch_sizes = []
val_loss = []

with open('Output_16870299.out', 'r') as file:
    output = file.read()

val_loss_pattern = r'(?<!loss: )val_loss:\s*(\d+\.\d+)'
val_loss = re.findall(r'Trial \d+ Complete.*?\nval_loss:\s*(\d+\.\d+)', output)
# Extract hyperparameters using regular expressions
learning_rate = re.findall(r'(\d+\.\d+e?-?\d*)\s*\|\s*\d+\.\d+\s*\|\s*learning_rate', output)
dropout = re.findall(r'(\d+\.\d+)\s*\|\s*\d+\.\d+\s*\|\s*dropout', output)
filter = re.findall(r'(\d+)\s*\|\s*\d+\s*\|\s*filter', output)
kernel_size_1 = re.findall(r'(\d+)\s*\|\s*\d+\s*\|\s*kernel_size_1', output)
kernel_size_2 = re.findall(r'(\d+)\s*\|\s*\d+\s*\|\s*kernel_size_2', output)
kernel_size_3 = re.findall(r'(\d+)\s*\|\s*\d+\s*\|\s*kernel_size_3', output)
activation_function = re.findall(r'(\w+)\s*\|\s*\w+\s*\|\s*act', output)
kernel_size_final = re.findall(r'(\d+)\s*\|\s*\d+\s*\|\s*kernel_size_final', output)
batch_size = re.findall(r'(\d+)\s*\|\s*\d+\s*\|\s*batch_size', output)

# Append the extracted values to the respective lists
val_loss.extend(val_loss)
learning_rates.extend(learning_rate)
dropouts.extend(dropout)
filters.extend(filter)
kernel_sizes_1.extend(kernel_size_1)
kernel_sizes_2.extend(kernel_size_2)
kernel_sizes_3.extend(kernel_size_3)
activation_functions.extend(activation_function)
kernel_size_finals.extend(kernel_size_final)
batch_sizes.extend(batch_size)

# Print the lists
print("Validation Loss:", [float(v) for v in val_loss])
print("Learning Rates:", [float(l) for l in learning_rates])
print("Dropouts:", [float(d) for d in dropouts])
print("Filters:", [int(i) for i in filters])
print("Kernel Sizes 1:", [int(i) for i in kernel_sizes_1])
print("Kernel Sizes 2:", [int(i) for i in kernel_sizes_2])
print("Kernel Sizes 3:", [int(i) for i in kernel_sizes_3])
print("Activation Functions:", activation_functions)
print("Kernel Size Finals:", [int(i) for i in kernel_size_finals])
print("Batch Sizes:", [int(i) for i in batch_sizes])

# %%
learning_rates = []
val_loss = []

with open('Output_16848366.out', 'r') as file:
    output = file.read()


# Extract hyperparameters using regular expressions
val_loss = re.findall(r'Trial \d+ Complete.*?\nval_loss:\s*(\d+\.\d+)', output)
learning_rate = re.findall(r'(\d+\.\d+e?-?\d*)\s*\|\s*\d+\.\d+\s*\|\s*learning_rate', output)

# Append the extracted values to the respective lists
val_loss.extend(val_loss)
learning_rates.extend(learning_rate)

# Print the lists
print("Validation Loss:", [float(v) for v in val_loss])
print("Learning Rates:", [float(l) for l in learning_rates])
# %%
val_loss_ = [0.0006964719586540014, 0.0008216162608005107, 0.0007146735093556345, 0.0006507922755554319, 0.0006726212229114025, 0.0007018773746676743, 0.0006725508300587535, 0.0006656494410708547, 0.002026051958091557, 0.0008483249903656542, 0.004001061972230673, 0.0006543729058466852, 0.0018023364758118986, 0.0006707748619373888, 0.0007953359698876739, 0.0007458931813016533, 0.0007106937724165618, 0.0006654819427058101, 0.0006791839306242764, 0.0007875422865618021, 0.000683556639123708, 0.0006867685494944453, 0.0007525262411218137, 0.0006627323594875634, 0.0006973577244207263, 0.0007166219665668905, 0.0007477700570598245, 0.0006430984230246394, 0.0007507112016901374, 0.0007077365834265947, 0.0008554863696917891, 0.0006537589826621116, 0.0007402214989997446, 0.000734008471481502, 0.0007566688547376543, 0.04991343513131142, 0.0007366381166502834, 0.0006470611644908786, 0.0006472721346653998, 0.000902340819593519, 0.0007221229438437149, 0.0008591192262247205, 0.000708099075127393]
lr = [0.00026441, 0.006098, 5.1015e-05, 0.002122, 0.00051596, 8.7057e-05, 0.00023866, 0.00035492, 0.0080493, 0.0045857, 0.0035545, 0.00010771, 0.0022565, 0.00086197, 0.0062381, 4.5655e-05, 0.00161, 0.00032117, 0.00033125, 1.3764e-05, 6.0381e-05, 0.00051373, 2.6543e-05, 0.00096568, 0.00031259, 3.5558e-05, 3.8242e-05, 0.0019385, 1.8666e-05, 0.00028879, 1.1815e-05, 0.00017432, 2.4134e-05, 1.9014e-05, 1.3357e-05, 0.0094637, 0.0024265, 0.0004308, 6.4966e-05, 0.01, 0.01, 1.0103e-05, 1.0304e-05]
val_loss = np.array(val_loss_)
val_loss = val_loss * (chl_max - chl_min) + chl_min
plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='red')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.grid()
plt.title('ConvLSTM w. SA \n Initial Optimization \n Learning Rate vs Validation Loss')
min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = min_val_loss * (chl_max - chl_min) + chl_min
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)

plt.axhline(min_val_loss, xmin=0, xmax=0.2, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.01, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr-0.001, min_val_loss+1.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

#%%
val_loss_ = [0.0006964719586540014, 0.0008216162608005107, 0.0007146735093556345, 0.0006507922755554319, 0.0006726212229114025, 0.0007018773746676743, 0.0006725508300587535, 0.0006656494410708547, 0.002026051958091557, 0.0008483249903656542, 0.004001061972230673, 0.0006543729058466852, 0.0018023364758118986, 0.0006707748619373888, 0.0007953359698876739, 0.0007458931813016533, 0.0007106937724165618, 0.0006654819427058101, 0.0006791839306242764, 0.0007875422865618021, 0.000683556639123708, 0.0006867685494944453, 0.0007525262411218137, 0.0006627323594875634, 0.0006973577244207263, 0.0007166219665668905, 0.0007477700570598245, 0.0006430984230246394, 0.0007507112016901374, 0.0007077365834265947, 0.0008554863696917891, 0.0006537589826621116, 0.0007402214989997446, 0.000734008471481502, 0.0007566688547376543, 0.04991343513131142, 0.0007366381166502834, 0.0006470611644908786, 0.0006472721346653998, 0.000902340819593519, 0.0007221229438437149, 0.0008591192262247205, 0.000708099075127393]
lr = [0.00026441, 0.006098, 5.1015e-05, 0.002122, 0.00051596, 8.7057e-05, 0.00023866, 0.00035492, 0.0080493, 0.0045857, 0.0035545, 0.00010771, 0.0022565, 0.00086197, 0.0062381, 4.5655e-05, 0.00161, 0.00032117, 0.00033125, 1.3764e-05, 6.0381e-05, 0.00051373, 2.6543e-05, 0.00096568, 0.00031259, 3.5558e-05, 3.8242e-05, 0.0019385, 1.8666e-05, 0.00028879, 1.1815e-05, 0.00017432, 2.4134e-05, 1.9014e-05, 1.3357e-05, 0.0094637, 0.0024265, 0.0004308, 6.4966e-05, 0.01, 0.01, 1.0103e-05, 1.0304e-05]
val_loss = np.array(val_loss_)
val_loss = val_loss * (chl_max - chl_min) + chl_min
plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='red')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.grid()
plt.xlim(0, 0.003)
plt.ylim(1.09, 1.35)
plt.title('ConvLSTM w. SA \n Initial Optimization \n Learning Rate vs Validation Loss')
min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = min_val_loss * (chl_max - chl_min) + chl_min
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)

plt.axhline(min_val_loss, xmin=0, xmax=0.67, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.1, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr-0.0015, min_val_loss+0.05),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
# %%
val_loss_ = [0.0006932114739902317, 0.0018492840370163321, 0.0006886056368239224, 0.0006511089764535427, 0.0007447635265998543, 0.0007572694239206612, 0.000664992043748498, 0.0006891482311766595, 0.0006642937287688255, 0.0006679086922667921, 0.0006606715789530427, 0.0007726194569841027, 0.0006763590482296423, 0.0006573969172313809, 0.0007005886500701308, 0.011824327036738396, 0.0007190803694538772, 0.0007198877539485693, 0.001840482451952994, 0.0006536774057894945, 0.0007902339042630046, 0.0006640149070881307, 0.0007127697067335248, 0.0007099519483745098, 0.0007232943223789334, 0.000737323712091893, 0.0007324890862219035, 0.0006866031442768871, 0.0007812797836959362, 0.0006693201535381377, 0.0006889959145337343, 0.0006832486414350569, 0.0007282810937613249, 0.0006810337759088725, 0.0008009769534692168, 0.000642020518425852, 0.0006492644583340735, 0.0006270294287241996, 0.0006879671942442656, 0.0007592648512218148, 0.0006694568623788655, 0.0006864911341108382, 0.0006908157607540488, 0.0006696952728088945]
lr = [0.0016021, 0.0081864, 0.00125, 0.00014499, 2.9612e-05, 1e-05, 0.00037448, 6.7705e-05, 0.00023415, 0.00064109, 0.00010191, 1.6559e-05, 0.00049218, 0.0001758, 0.00012637, 0.0027012, 4.4419e-05, 2.2077e-05, 0.01, 0.00087592, 1.2652e-05, 0.00029607, 5.4834e-05, 8.3186e-05, 3.6118e-05, 0.0014347, 0.001037, 0.00074698, 1.919e-05, 0.00020359, 0.00042904, 0.00056315, 2.5577e-05, 0.00033153, 1.112e-05, 0.000263, 6.111e-05, 9.2709e-05, 4.01e-05 , 1.4523e-05, 0.00015936,  0.0011498 , 0.00081599, 0.00011286]
val_loss = np.array(val_loss_)
val_loss = val_loss * (chl_max - chl_min) + chl_min
print(len(val_loss), len(lr))
plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='blue')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.title('ConvLSTM w. SA \n logistic learning rate sampling refinement \n Learning Rate vs Validation Loss')
plt.xlim(0, 0.002)
plt.ylim(1.1, 1.15)
min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = min_val_loss * (chl_max - chl_min) + chl_min
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)

plt.axhline(min_val_loss, xmin=0, xmax=0.07, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.17, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr+0.0001, min_val_loss+0.023),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

#%%

val_loss_ = [0.0006932114739902317, 0.0018492840370163321, 0.0006886056368239224, 0.0006511089764535427, 0.0007447635265998543, 0.0007572694239206612, 0.000664992043748498, 0.0006891482311766595, 0.0006642937287688255, 0.0006679086922667921, 0.0006606715789530427, 0.0007726194569841027, 0.0006763590482296423, 0.0006573969172313809, 0.0007005886500701308, 0.011824327036738396, 0.0007190803694538772, 0.0007198877539485693, 0.001840482451952994, 0.0006536774057894945, 0.0007902339042630046, 0.0006640149070881307, 0.0007127697067335248, 0.0007099519483745098, 0.0007232943223789334, 0.000737323712091893, 0.0007324890862219035, 0.0006866031442768871, 0.0007812797836959362, 0.0006693201535381377, 0.0006889959145337343, 0.0006832486414350569, 0.0007282810937613249, 0.0006810337759088725, 0.0008009769534692168, 0.000642020518425852, 0.0006492644583340735, 0.0006270294287241996, 0.0006879671942442656, 0.0007592648512218148, 0.0006694568623788655, 0.0006864911341108382, 0.0006908157607540488, 0.0006696952728088945]
lr = [0.0016021, 0.0081864, 0.00125, 0.00014499, 2.9612e-05, 1e-05, 0.00037448, 6.7705e-05, 0.00023415, 0.00064109, 0.00010191, 1.6559e-05, 0.00049218, 0.0001758, 0.00012637, 0.0027012, 4.4419e-05, 2.2077e-05, 0.01, 0.00087592, 1.2652e-05, 0.00029607, 5.4834e-05, 8.3186e-05, 3.6118e-05, 0.0014347, 0.001037, 0.00074698, 1.919e-05, 0.00020359, 0.00042904, 0.00056315, 2.5577e-05, 0.00033153, 1.112e-05, 0.000263, 6.111e-05, 9.2709e-05, 4.01e-05 , 1.4523e-05, 0.00015936,  0.0011498 , 0.00081599, 0.00011286]
val_loss = np.array(val_loss_)
val_loss = val_loss * (chl_max - chl_min) + chl_min
print(len(val_loss), len(lr))
plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='blue')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.title('ConvLSTM w. SA \n logistic learning rate sampling refinement \n Learning Rate vs Validation Loss')
# plt.xlim(0, 0.002)
# plt.ylim(0.0005, 0.001)
min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = min_val_loss * (chl_max - chl_min) + chl_min
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)

plt.axhline(min_val_loss, xmin=0, xmax=0.07, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.07, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr+0.0004, min_val_loss+0.3),
             arrowprops=dict(facecolor='black', arrowstyle='->'))# %%
# %%

val_loss_ =  [0.001851619048975408, 0.001888351608067751, 0.002667052582837641, 0.0019421172211878002, 0.0018881483376026154, 0.0007720913086086511, 0.0007405377295799554, 0.00067883497744333, 0.0006968418043106794, 0.0019210136588662862, 0.0006634156068321317, 0.002064406406134367, 0.0006472776993177831, 0.0006589053652714938, 0.0006745514483191073, 0.0006581702921539545, 0.000651697147404775, 0.001878354293294251, 0.0006540565425530076, 0.0006686998088844121, 0.0022341293515637517, 0.0018550362065434456, 0.0006550406815949828, 0.001880033640190959, 0.0006679598160553723, 0.000686745154671371, 0.0007066698628477753, 0.0006464158929884434, 0.0006816101190634072, 0.0018529522931203246, 0.0010244838590733707, 0.000649299225769937, 0.0006713835150003434, 0.0006763767101801932, 0.0006609754730015993, 0.0007080187136307359, 0.0006462023290805519, 0.0006486583664081991, 0.0006506289704702794, 0.0006598811293952167]
lr = [0.00803, 0.00513, 0.00337, 0.00837, 0.00653, 0.00043, 0.00057, 0.00999, 0.00033, 0.00155, 0.00021, 0.00023, 0.00025, 0.00017, 0.00017, 0.00727, 0.00017, 0.00069, 0.00435, 0.00923, 0.00015, 0.00583, 0.00019, 0.00021, 0.00071, 0.00021, 0.00023, 0.00243, 0.00103, 0.00013, 0.00013, 0.00033, 0.00015, 0.00015, 0.00019, 0.00019, 0.00019, 0.00019, 0.00019, 0.00019]
val_loss = np.array(val_loss_)
val_loss = val_loss * (chl_max - chl_min) + chl_min
print(len(val_loss), len(lr))
plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='green')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.title('ConvLSTM w. SA \n normal learning rate sampling refinement \n Learning Rate vs Validation Loss')
# plt.xlim(0, 0.002)
# plt.ylim(1.09, 1.5)
min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = min_val_loss * (chl_max - chl_min) + chl_min
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)

plt.axhline(min_val_loss, xmin=0, xmax=0.07, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.07, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr+0.0004, min_val_loss+0.05),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
#%%

val_loss_ = [0.11534601703286171, 0.15100119829177858, 0.1481168645620346, 0.13530903398990632, 0.10545781522989273, 0.12907760381698608, 0.1202102243900299, 0.11732859939336776, 0.11806652545928956, 0.14362720489501954, 0.18014955401420593, 0.10689112305641174, 0.11641745656728744, 0.1116730760037899, 0.17525187849998475, 0.13447513222694396, 0.10842568099498749, 0.13976158142089845, 0.12348813593387603, 0.1212417510151863, 0.325687849521637, 0.1576099395751953, 0.11275146961212158, 0.1552150970697403, 0.11658831179141999, 1.1850651025772094, 0.17044214010238648, 0.11412149727344513, 0.1109993976354599, 0.11637912392616272, 0.12101535737514496, 0.11029252618551254, 0.13862656444311142, 0.11188330292701722, 0.10752026140689849, 0.1200150865316391, 0.1314651796221733, 0.1066814860701561]
lr = [5.9833e-05 , 1.3933e-05, 0.0017321 , 1.1733e-05, 0.00048596, 8.1058e-05, 1.9688e-05, 0.00010602, 0.00090274, 4.1244e-05, 1.9277e-05, 0.00034728, 0.001947, 5.0763e-05, 0.0096012, 4.2772e-05, 0.00014373, 4.973e-05, 7.4302e-05, 0.0031885, 0.0076814, 2.2609e-05, 0.00076242, 3.3895e-05, 0.00026612, 0.0044691, 0.0049878, 0.00012072, 0.00018983, 0.0011292, 2.1488e-05, 0.00045069, 1.4688e-05, 5.7124e-05, 0.00025175, 0.0022051, 1.2253e-05, 0.00011609]
val_loss = np.exp(np.array(val_loss_))
print(len(val_loss), len(lr))
plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='red')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.grid()
plt.xlim(0, 0.0025)
plt.ylim(1.09, 1.25)
plt.title('ConvLSTM \n Initial Optimization \n Learning Rate vs Validation Loss')


min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = np.exp(min_val_loss)
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)

plt.axhline(min_val_loss, xmin=0, xmax=0.19, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.1, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr-0.00002, min_val_loss+0.058),
             arrowprops=dict(facecolor='black', arrowstyle='->'))# %%

#%%
val_loss_ = [0.11534601703286171, 0.15100119829177858, 0.1481168645620346, 0.13530903398990632, 0.10545781522989273, 0.12907760381698608, 0.1202102243900299, 0.11732859939336776, 0.11806652545928956, 0.14362720489501954, 0.18014955401420593, 0.10689112305641174, 0.11641745656728744, 0.1116730760037899, 0.17525187849998475, 0.13447513222694396, 0.10842568099498749, 0.13976158142089845, 0.12348813593387603, 0.1212417510151863, 0.325687849521637, 0.1576099395751953, 0.11275146961212158, 0.1552150970697403, 0.11658831179141999, 1.1850651025772094, 0.17044214010238648, 0.11412149727344513, 0.1109993976354599, 0.11637912392616272, 0.12101535737514496, 0.11029252618551254, 0.13862656444311142, 0.11188330292701722, 0.10752026140689849, 0.1200150865316391, 0.1314651796221733, 0.1066814860701561]
lr = [5.9833e-05 , 1.3933e-05, 0.0017321 , 1.1733e-05, 0.00048596, 8.1058e-05, 1.9688e-05, 0.00010602, 0.00090274, 4.1244e-05, 1.9277e-05, 0.00034728, 0.001947, 5.0763e-05, 0.0096012, 4.2772e-05, 0.00014373, 4.973e-05, 7.4302e-05, 0.0031885, 0.0076814, 2.2609e-05, 0.00076242, 3.3895e-05, 0.00026612, 0.0044691, 0.0049878, 0.00012072, 0.00018983, 0.0011292, 2.1488e-05, 0.00045069, 1.4688e-05, 5.7124e-05, 0.00025175, 0.0022051, 1.2253e-05, 0.00011609]
val_loss = np.exp(np.array(val_loss_))
print(len(val_loss), len(lr))
plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='red')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.grid()

plt.title('ConvLSTM \n Initial Optimization \n Learning Rate vs Validation Loss')


min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = np.exp(min_val_loss)
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)

plt.axhline(min_val_loss, xmin=0, xmax=0.1, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.07, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr-0.00002, min_val_loss+0.6),
             arrowprops=dict(facecolor='black', arrowstyle='->'))# %%

# %%
val_loss_ = [0.16833260118961335, 0.12542222678661347, 0.11548763155937194, 0.1331148052215576, 0.16756903290748595, 0.13997053861618042, 0.14975825279951097, 0.13950595676898955, 0.1302020889520645, 0.12311346173286437, 0.12178056806325913, 0.13112042874097823, 0.14510686576366424, 0.13119987428188323, 0.12264980733394623, 0.14065954685211182, 0.1281343385577202, 0.1303410002589226, 0.12324999511241913, 0.11007740914821625, 0.11897000551223755, 0.12439891457557678, 0.1086695882678032, 0.11212838470935821, 0.10743661671876907, 0.14907063245773317, 0.1304983013868332, 0.12584250628948213, 0.13052306830883026, 0.12785954594612123, 0.12981321692466735, 0.144487224817276, 0.1256083595752716, 0.12668040335178377, 0.1362404552102089, 0.14018715620040895, 0.15709981679916382]
lr = [0.00853, 0.00461, 0.00063, 0.00765, 0.00927, 0.00163, 0.00131, 0.00519, 0.00393, 0.00319, 0.00667, 0.00255, 0.00593, 0.00353, 0.00717, 0.00083, 0.00427, 0.00999, 0.00043, 0.00629, 0.00489, 0.00049, 0.00047, 0.00029, 0.00371, 0.00335, 0.00297, 0.00977, 0.00645, 0.00613, 0.00555, 0.00205, 0.00801, 0.00407, 0.00221, 0.00187, 0.00817]
print(len(val_loss), len(lr))
val_loss = np.exp(np.array(val_loss_))
min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = np.exp(min_val_loss)

plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='green')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.grid()
# plt.xlim(0, 0.0025)
# plt.ylim(0.1, 0.2)
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)
plt.title('ConvLSTM \n normal learning rate sampling refinement \n Learning Rate vs Validation Loss')

plt.axhline(min_val_loss, xmin=0, xmax=0.37, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.07, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr-0.005, min_val_loss-0.04),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
#%%
val_loss_ = [0.16833260118961335, 0.12542222678661347, 0.11548763155937194, 0.1331148052215576, 0.16756903290748595, 0.13997053861618042, 0.14975825279951097, 0.13950595676898955, 0.1302020889520645, 0.12311346173286437, 0.12178056806325913, 0.13112042874097823, 0.14510686576366424, 0.13119987428188323, 0.12264980733394623, 0.14065954685211182, 0.1281343385577202, 0.1303410002589226, 0.12324999511241913, 0.11007740914821625, 0.11897000551223755, 0.12439891457557678, 0.1086695882678032, 0.11212838470935821, 0.10743661671876907, 0.14907063245773317, 0.1304983013868332, 0.12584250628948213, 0.13052306830883026, 0.12785954594612123, 0.12981321692466735, 0.144487224817276, 0.1256083595752716, 0.12668040335178377, 0.1362404552102089, 0.14018715620040895, 0.15709981679916382]
lr = [0.00853, 0.00461, 0.00063, 0.00765, 0.00927, 0.00163, 0.00131, 0.00519, 0.00393, 0.00319, 0.00667, 0.00255, 0.00593, 0.00353, 0.00717, 0.00083, 0.00427, 0.00999, 0.00043, 0.00629, 0.00489, 0.00049, 0.00047, 0.00029, 0.00371, 0.00335, 0.00297, 0.00977, 0.00645, 0.00613, 0.00555, 0.00205, 0.00801, 0.00407, 0.00221, 0.00187, 0.00817]
print(len(val_loss), len(lr))
val_loss = np.exp(np.array(val_loss_))
min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = np.exp(min_val_loss)

plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='green')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.grid()
plt.xlim(0, 0.004)
plt.ylim(1.09, 1.3)
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)
plt.title('ConvLSTM \n normal learning rate sampling refinement \n Learning Rate vs Validation Loss')

plt.axhline(min_val_loss, xmin=0, xmax=0.93, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.13, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr-0.0025, min_val_loss+0.08),
             arrowprops=dict(facecolor='black', arrowstyle='->'))# %%
# %%
val_loss_ = [0.11870896875858307, 0.11581842541694641, 0.11468343496322632, 0.10220532774925233, 0.10789603233337403, 0.10666180014610291, 0.11719968259334564, 0.1576881742477417, 0.12420464485883713, 0.10733043611049652, 0.11194095134735108, 0.10786906719207763, 0.10595948040485383, 0.12548853754997252, 0.10720243692398071, 0.11116238415241242, 0.11506392121315002, 0.11378388166427612, 0.10768462359905243, 0.10166754990816117, 0.10497631013393402, 0.10427423506975174, 0.10367191284894943, 0.1047658383846283, 0.1132063889503479, 0.1073923310637474, 0.12666390627622603, 0.10906744092702865, 0.10566156536340714, 0.09996436417102814, 0.10700424879789353, 0.10952079862356186, 0.1045716518163681, 0.11197208046913147, 0.1282346147298813, 0.10535947889089585]
lr = [0.0035341, 0.00042471, 0.00094474, 0.00016412, 0.00018202, 0.00014505, 0.00016566, 0.0028275, 0.00096225, 0.00016272, 0.00018547, 0.00014246, 0.00014804, 0.00082397, 0.00017819, 0.0001914, 0.00013932, 0.0001509, 0.00021702, 0.00021289, 0.00019666, 0.00020021, 0.00020934, 0.00020469, 0.00090643, 0.00022356, 0.0099411, 0.00017402, 0.00022915, 0.0002579, 0.00023441, 0.00025372, 0.00026212, 0.00030734, 0.0054734, 0.00026944]
print(len(val_loss), len(lr))

val_loss = np.exp(np.array(val_loss_))
min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = np.exp(min_val_loss)

plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='blue')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.grid()
# plt.xlim(0, 0.0025)
# plt.ylim(0.1, 0.2)
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)
plt.title('ConvLSTM \n logistic learning rate sampling refinement \n Learning Rate vs Validation Loss')

plt.axhline(min_val_loss, xmin=0, xmax=0.07, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.07, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr+0.000, min_val_loss+0.037),
             arrowprops=dict(facecolor='black', arrowstyle='->'))# %%
# %%
val_loss_ = [0.11870896875858307, 0.11581842541694641, 0.11468343496322632, 0.10220532774925233, 0.10789603233337403, 0.10666180014610291, 0.11719968259334564, 0.1576881742477417, 0.12420464485883713, 0.10733043611049652, 0.11194095134735108, 0.10786906719207763, 0.10595948040485383, 0.12548853754997252, 0.10720243692398071, 0.11116238415241242, 0.11506392121315002, 0.11378388166427612, 0.10768462359905243, 0.10166754990816117, 0.10497631013393402, 0.10427423506975174, 0.10367191284894943, 0.1047658383846283, 0.1132063889503479, 0.1073923310637474, 0.12666390627622603, 0.10906744092702865, 0.10566156536340714, 0.09996436417102814, 0.10700424879789353, 0.10952079862356186, 0.1045716518163681, 0.11197208046913147, 0.1282346147298813, 0.10535947889089585]
lr = [0.0035341, 0.00042471, 0.00094474, 0.00016412, 0.00018202, 0.00014505, 0.00016566, 0.0028275, 0.00096225, 0.00016272, 0.00018547, 0.00014246, 0.00014804, 0.00082397, 0.00017819, 0.0001914, 0.00013932, 0.0001509, 0.00021702, 0.00021289, 0.00019666, 0.00020021, 0.00020934, 0.00020469, 0.00090643, 0.00022356, 0.0099411, 0.00017402, 0.00022915, 0.0002579, 0.00023441, 0.00025372, 0.00026212, 0.00030734, 0.0054734, 0.00026944]
print(len(val_loss), len(lr))


val_loss = np.exp(np.array(val_loss_))
min_val_loss = min(val_loss_)
min_lr = lr[val_loss_.index(min_val_loss)]
min_val_loss = np.exp(min_val_loss)

plt.figure(figsize=(4, 2))
plt.scatter(lr, val_loss, color='blue')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.grid()
plt.xlim(0, 0.001)
plt.ylim(1.1, 1.14)
plt.scatter([min_lr], [min_val_loss], color='yellow', edgecolors='black', s=50, zorder=9)
plt.title('ConvLSTM \n logistic learning rate sampling refinement \n Learning Rate vs Validation Loss')

plt.axhline(min_val_loss, xmin=0, xmax=0.27, color='black', linestyle='dashed')
plt.axvline(min_lr, ymin=0, ymax=0.15, color='black', linestyle='dashed')

plt.annotate(f"Optimal point \nlearning rate: {min_lr} \nvalidation loss: {min_val_loss:.6f}", 
             xy=(min_lr, min_val_loss), xytext=(min_lr+0.0001, min_val_loss-0.001),
             arrowprops=dict(facecolor='black', arrowstyle='->'))# %%
# %%
# Plots of feature importance with backward feature selection
####### GULF OF RIGA #######
baselines = [3.0465, 3.2528, 3.4136, 3.6419, 3.6796]

COLS = ['Chlorophyll-a','Cloudmask', 'Latitude', 'Radiation', 'Sea surface temperature', 'Cosine of day of year', 'Air temperature',
            'Sine of day of year', 'Wind v10', 'Precipitation', 'Wave VHM0', 'Topography', 'Longitude', 'Wave VMDR', 'Wind u10']


results_1_ = [3.7099, 3.7099, 3.7099, 3.1150, 3.1150, 3.1150, 3.1150, 3.0594, 3.2268, 3.0657, 3.0657, 3.0598, 3.2862, 3.0020, 3.0020]
results_2_ = [3.7248, 3.7248, 3.7248, 3.4463, 3.4463, 3.4463, 3.4463, 3.2309, 3.3828, 3.3931, 3.3931, 3.2943, 3.4360, 3.2065, 3.2065]
results_3_ = [3.7219, 3.7219, 3.7219, 3.4691, 3.4691, 3.4691, 3.4691, 3.3636, 3.3919, 3.5518, 3.5518, 3.4612, 3.5842, 3.3792, 3.3792]
results_4_ = [3.8258, 3.8258, 3.8258, 3.7196, 3.7196, 3.7196, 3.7196, 3.6076, 3.6266, 3.7122, 3.7122, 3.6733, 3.7548, 3.6387, 3.6387]
results_5_ = [3.7556, 3.7556, 3.7556, 3.7518, 3.7518, 3.7518, 3.7518, 3.5998, 3.6124, 3.7724, 3.7724, 3.7126, 3.8197, 3.6951, 3.6951]

cm = plt.cm.get_cmap('viridis')
my_cmap = cm(np.linspace(0,1,15))

# Create new colormap
my_cmap = ListedColormap(my_cmap)

colors = [my_cmap(0), my_cmap(0), my_cmap(0), my_cmap(12), my_cmap(2), my_cmap(2), my_cmap(2), my_cmap(2), my_cmap(8), my_cmap(8), my_cmap(6), my_cmap(10), 'red', my_cmap(14), my_cmap(14), my_cmap(4)]

results_1 = []
results_2 = []
results_3 = []
results_4 = []
results_5 = []

results_1.append({'feature':'BASELINE','rmses':baselines[0]})    
results_2.append({'feature':'BASELINE','rmses':baselines[1]}) 
results_3.append({'feature':'BASELINE','rmses':baselines[2]}) 
results_4.append({'feature':'BASELINE','rmses':baselines[3]})    
results_5.append({'feature':'BASELINE','rmses':baselines[4]}) 

for k in range(len(COLS)):
    results_1.append({'feature':COLS[k],'rmses':results_1_[k]})
    results_2.append({'feature':COLS[k],'rmses':results_2_[k]})
    results_3.append({'feature':COLS[k],'rmses':results_3_[k]})
    results_4.append({'feature':COLS[k],'rmses':results_4_[k]})
    results_5.append({'feature':COLS[k],'rmses':results_5_[k]})
    
results_1 = pd.DataFrame(results_1).sort_values(by='rmses',ascending=True)
results_2 = pd.DataFrame(results_2).sort_values(by='rmses',ascending=True)
results_3 = pd.DataFrame(results_3).sort_values(by='rmses',ascending=True)
results_4 = pd.DataFrame(results_4).sort_values(by='rmses',ascending=True)
results_5 = pd.DataFrame(results_5).sort_values(by='rmses',ascending=True)

results = {
    0: results_1,
    1: results_2,
    2: results_3,
    3: results_4,
    4: results_5
}


fig, axes = plt.subplots(1,5,figsize=(22,6))
for i in range(5):
    df = results[i]
    print(df)
    axes[i].barh(np.arange(len(COLS)+1),df.rmses, color="seagreen")
    axes[i].set_yticks(np.arange(len(COLS)+1),df.feature.values, size=12)
    axes[i].set_title(f'Day {i+1}',size=16)
    axes[i].set_ylim((-1,len(COLS)+1))
    axes[i].plot([baselines[i],baselines[i]],[-1,len(COLS)+1], '--', color='red',
                label=f'Baseline \nRMSE={baselines[i]:.3f}')
    axes[i].set_xlabel(f'RMSE',size=14)
    axes[i].legend(fontsize=11)
    axes[i].tick_params(axis='both', which='major', labelsize=12)
    axes[i].tick_params(axis='both', which='minor', labelsize=12)

axes[0].set_xlim((2.5,4))
axes[1].set_xlim((2.5,4))
axes[2].set_xlim((2.5,4))
axes[3].set_xlim((2.5,4))
axes[4].set_xlim((2.5,4))    
axes[0].set_ylabel('Permuted Feature',size=14)

plt.tight_layout()
plt.show()

#%%
# get the total feature importance for each feature
result = pd.concat(list(results.values()), axis=1)
result_ = result.drop(columns=['feature'])
result_ = result_.mean(axis=1)
result = result.loc[:, ~result.columns.duplicated()]
result['total_rmse'] = result_
result = result.sort_values(by='total_rmse',ascending=True)
print(result_)
print(result)

baseline = result[result.feature=='BASELINE']['total_rmse'].values[0]
print(baseline)

fix, axes = plt.subplots(1,1,figsize=(6,6))
print(result.total_rmse.values)
# reverse the colors list
colors = colors[::-1]
print(len(colors))
print(len(result.total_rmse.values))
axes.barh(np.arange(len(COLS)+1),result.total_rmse.values, color=colors)
axes.set_yticks(np.arange(len(COLS)+1),result.feature.values, size=12)
axes.set_title(f'Total Feature Importance',size=16)
axes.plot([baseline,baseline],
             [-1,len(COLS)+1], '--', color='red', label=f'Baseline \nRMSE={baseline:.3f}')
axes.set_ylim((-1,len(COLS)+1))
axes.set_xlabel(f'RMSE',size=14)
axes.tick_params(axis='both', which='major', labelsize=12)
axes.tick_params(axis='both', which='minor', labelsize=12)
axes.set_xlim((2.5,4))
axes.legend(fontsize=11)
plt.tight_layout()
plt.show()

#%%
from matplotlib.colors import ListedColormap
cm = plt.cm.get_cmap('viridis')
my_cmap = cm(np.linspace(0,1,15))

# Create new colormap
my_cmap = ListedColormap(my_cmap)
#plot a time series for each feature group
groups = np.empty((5,len(COLS)))

for k in range(len(COLS)):
    groups[0,k]=results_1_[k]
    groups[1,k]=results_2_[k]
    groups[2,k]=results_3_[k]
    groups[3,k]=results_4_[k]
    groups[4,k]=results_5_[k]
    
    
plt.figure(figsize=(6,4))
for i in range(len(COLS)):
    plt.plot(groups[:,i],label=COLS[i], color=colors[i])

plt.plot(baselines,label='Baseline', color='red', linestyle='--')
plt.ylabel('RMSE')
plt.xticks(np.arange(5),['Day 1','Day 2','Day 3','Day 4','Day 5'])
# put the legend next to the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    
plt.show()

    


# %%

####### GOTLAND BASIN #######
baselines = [1.1196, 1.0241, 1.0892, 1.1814, 1.2819]

COLS = ['Chlorophyll-a','Cloudmask', 'Sea surface temperature', 'Sine of day of year', 'Air temperature',
            'Radiation', 'Cosine of day of year', 'Precipitation', 'Wind v10',  'Wave VHM0', 'Wind u10', 'Topography', 'Longitude', 'Wave VMDR', 'Latitude']


results_1_ = [1.5839, 1.5839, 1.2520, 1.2520, 1.2520, 1.0941, 1.0941, 1.0941, 1.0940, 1.1799, 1.1799, 1.0776, 1.0776, 1.3803, 1.0859]
results_2_ = [1.5698, 1.5698, 1.3144, 1.3144, 1.3144, 1.1205, 1.1205, 1.1205, 1.1732, 1.2413, 1.2413, 0.9725, 0.9725, 1.3928, 0.9929]
results_3_ = [1.6318, 1.6318, 1.4326, 1.4326, 1.4326, 1.1945, 1.1945, 1.1945, 1.2059, 1.3121, 1.3121, 1.0653, 1.0653, 1.2989, 1.0825]
results_4_ = [1.5342, 1.5342, 1.3679, 1.3679, 1.3679, 1.2338, 1.2338, 1.2338, 1.1504, 1.2111, 1.2111, 1.1169, 1.1169, 1.2856, 1.1151]
results_5_ = [1.7150, 1.7150, 1.4016, 1.4016, 1.4016, 1.3491, 1.3491, 1.3491, 1.2927, 1.3454, 1.3454, 1.2431, 1.2431, 1.4530, 1.2498]

colors = [my_cmap(0), my_cmap(0), my_cmap(12), my_cmap(2), my_cmap(2), my_cmap(2), my_cmap(8), my_cmap(8), my_cmap(4), my_cmap(4), my_cmap(4), my_cmap(6), 'red', my_cmap(14), my_cmap(10), my_cmap(10)]

results_1 = []
results_2 = []
results_3 = []
results_4 = []
results_5 = []

results_1.append({'feature':'BASELINE','rmses':baselines[0]})    
results_2.append({'feature':'BASELINE','rmses':baselines[1]}) 
results_3.append({'feature':'BASELINE','rmses':baselines[2]}) 
results_4.append({'feature':'BASELINE','rmses':baselines[3]})    
results_5.append({'feature':'BASELINE','rmses':baselines[4]}) 

for k in range(len(COLS)):
    results_1.append({'feature':COLS[k],'rmses':results_1_[k]})
    results_2.append({'feature':COLS[k],'rmses':results_2_[k]})
    results_3.append({'feature':COLS[k],'rmses':results_3_[k]})
    results_4.append({'feature':COLS[k],'rmses':results_4_[k]})
    results_5.append({'feature':COLS[k],'rmses':results_5_[k]})
    
results_1 = pd.DataFrame(results_1).sort_values(by='rmses',ascending=True)
results_2 = pd.DataFrame(results_2).sort_values(by='rmses',ascending=True)
results_3 = pd.DataFrame(results_3).sort_values(by='rmses',ascending=True)
results_4 = pd.DataFrame(results_4).sort_values(by='rmses',ascending=True)
results_5 = pd.DataFrame(results_5).sort_values(by='rmses',ascending=True)

results = {
    0: results_1,
    1: results_2,
    2: results_3,
    3: results_4,
    4: results_5
}


fig, axes = plt.subplots(1,5,figsize=(22,6))
for i in range(5):
    df = results[i]
    print(df)
    axes[i].barh(np.arange(len(COLS)+1),df.rmses, color="seagreen")
    axes[i].set_yticks(np.arange(len(COLS)+1),df.feature.values, size=12)
    axes[i].set_title(f'Day {i+1}',size=16)
    axes[i].set_ylim((-1,len(COLS)+1))
    axes[i].plot([baselines[i],baselines[i]],[-1,len(COLS)+1], '--', color='red',
                label=f'Baseline \nRMSE={baselines[i]:.3f}')
    axes[i].set_xlabel(f'RMSE',size=14)
    axes[i].legend(fontsize=11)
    axes[i].tick_params(axis='both', which='major', labelsize=12)
    axes[i].tick_params(axis='both', which='minor', labelsize=12)

axes[0].set_xlim((0.9,2))
axes[1].set_xlim((0.9,2))
axes[2].set_xlim((0.9,2))
axes[3].set_xlim((0.9,2))
axes[4].set_xlim((0.9,2))    
axes[0].set_ylabel('Permuted Feature',size=14)

plt.tight_layout()
plt.show()

#%%
# get the total feature importance for each feature
result = pd.concat(list(results.values()), axis=1)
result_ = result.drop(columns=['feature'])
result_ = result_.mean(axis=1)
result = result.loc[:, ~result.columns.duplicated()]
result['total_rmse'] = result_
result = result.sort_values(by='total_rmse',ascending=True)
print(result_)
print(result)

baseline = result[result.feature=='BASELINE']['total_rmse'].values[0]
print(baseline)
colors = colors[::-1]
fix, axes = plt.subplots(1,1,figsize=(6,6))
axes.barh(np.arange(len(COLS)+1),result.total_rmse.values, color=colors)
axes.set_yticks(np.arange(len(COLS)+1),result.feature.values, size=12)
axes.set_title(f'Total Feature Importance',size=16)
axes.plot([baseline,baseline],
             [-1,len(COLS)+1], '--', color='red', label=f'Baseline \nRMSE={baseline:.3f}')
axes.set_ylim((-1,len(COLS)+1))
axes.set_xlabel(f'RMSE',size=14)
axes.tick_params(axis='both', which='major', labelsize=12)
axes.tick_params(axis='both', which='minor', labelsize=12)
axes.set_xlim((1,2))
axes.legend(fontsize=11)
plt.tight_layout()
plt.show()

#%%
from matplotlib.colors import ListedColormap
cm = plt.cm.get_cmap('viridis')
my_cmap = cm(np.linspace(0,1,15))

# Create new colormap
my_cmap = ListedColormap(my_cmap)
#plot a time series for each feature group
groups = np.empty((5,8))

for k in range(len(COLS)):
    groups[0,k]=results_1_[k]
    groups[1,k]=results_2_[k]
    groups[2,k]=results_3_[k]
    groups[3,k]=results_4_[k]
    groups[4,k]=results_5_[k]
    
plt.figure(figsize=(6,4))
plt.plot(groups[:,0],label='Group 1', color=my_cmap(2))
plt.plot(groups[:,1],label='Group 2', color=my_cmap(4))
plt.plot(groups[:,2],label='Group 3', color=my_cmap(5))
plt.plot(groups[:,3],label='Group 4', color=my_cmap(6))
plt.plot(groups[:,4],label='Group 5', color=my_cmap(8))
plt.plot(groups[:,5],label='Group 6', color=my_cmap(10))
plt.plot(groups[:,6],label='Group 7', color=my_cmap(12))
plt.plot(groups[:,7],label='Group 8', color=my_cmap(14))
plt.plot(baselines,label='Baseline', color='red', linestyle='--')
plt.ylabel('RMSE')
plt.xticks(np.arange(5),['Day 1','Day 2','Day 3','Day 4','Day 5'])
plt.ylim((0.5,1.8))
plt.legend()
plt.show()

#%%
####### SOUTHERN KATTEGAT #######
baselines = [1.4431, 1.4536, 1.4671, 1.4644, 1.4670]

COLS = ['Chlorophyll-a','Cloudmask', 'Topography', 'Radiation', 'Sea surface temperature', 'Cosine of day of year', 'Air temperature',
            'Sine of day of year', 'Wind v10', 'Precipitation', 'Wave VHM0', 'Latitude', 'Longitude', 'Wave VMDR', 'Wind u10']

colors = [my_cmap(0), my_cmap(0), my_cmap(0), my_cmap(8), my_cmap(8), 'red', my_cmap(6), my_cmap(4), my_cmap(10), my_cmap(2), my_cmap(2), my_cmap(2), my_cmap(2), my_cmap(10), my_cmap(14), my_cmap(14)]
results_1_ = [2.4749, 2.4749, 2.4749, 1.3004, 1.3004, 1.3004, 1.3004, 1.3665, 1.3593, 1.5293, 1.5293, 1.3292, 1.3382, 1.3390, 1.3390]
results_2_ = [2.4800, 2.4800, 2.4800, 1.3519, 1.3519, 1.3519, 1.3519, 1.3804, 1.4012, 1.4721, 1.4721, 1.3746, 1.3632, 1.3005, 1.3005]
results_3_ = [2.5249, 2.5249, 2.5249, 1.4185, 1.4185, 1.4185, 1.4185, 1.4347, 1.4684, 1.5398, 1.5398, 1.4264, 1.4221, 1.3623, 1.3623]
results_4_ = [2.5186, 2.5186, 2.5186, 1.4082, 1.4082, 1.4082, 1.4082, 1.4259, 1.4104, 1.5315, 1.5315, 1.4384, 1.3724, 1.3385, 1.3385]
results_5_ = [2.5041, 2.5041, 2.5041, 1.4627, 1.4627, 1.4627, 1.4627, 1.4744, 1.4590, 1.5577, 1.5577, 1.4929, 1.4077, 1.4219, 1.4219]

results_1 = []
results_2 = []
results_3 = []
results_4 = []
results_5 = []

results_1.append({'feature':'BASELINE','rmses':baselines[0]})    
results_2.append({'feature':'BASELINE','rmses':baselines[1]}) 
results_3.append({'feature':'BASELINE','rmses':baselines[2]}) 
results_4.append({'feature':'BASELINE','rmses':baselines[3]})    
results_5.append({'feature':'BASELINE','rmses':baselines[4]}) 

for k in range(len(COLS)):
    results_1.append({'feature':COLS[k],'rmses':results_1_[k]})
    results_2.append({'feature':COLS[k],'rmses':results_2_[k]})
    results_3.append({'feature':COLS[k],'rmses':results_3_[k]})
    results_4.append({'feature':COLS[k],'rmses':results_4_[k]})
    results_5.append({'feature':COLS[k],'rmses':results_5_[k]})
    
results_1 = pd.DataFrame(results_1).sort_values(by='rmses',ascending=True)
results_2 = pd.DataFrame(results_2).sort_values(by='rmses',ascending=True)
results_3 = pd.DataFrame(results_3).sort_values(by='rmses',ascending=True)
results_4 = pd.DataFrame(results_4).sort_values(by='rmses',ascending=True)
results_5 = pd.DataFrame(results_5).sort_values(by='rmses',ascending=True)

results = {
    0: results_1,
    1: results_2,
    2: results_3,
    3: results_4,
    4: results_5
}


fig, axes = plt.subplots(1,5,figsize=(22,6))
for i in range(5):
    df = results[i]
    print(df)
    axes[i].barh(np.arange(len(COLS)+1),df.rmses, color="seagreen")
    axes[i].set_yticks(np.arange(len(COLS)+1),df.feature.values, size=12)
    axes[i].set_title(f'Day {i+1}',size=16)
    axes[i].set_ylim((-1,len(COLS)+1))
    axes[i].plot([baselines[i],baselines[i]],[-1,len(COLS)+1], '--', color='red',
                label=f'Baseline \nRMSE={baselines[i]:.3f}')
    axes[i].set_xlabel(f'RMSE',size=14)
    axes[i].legend(fontsize=11)
    axes[i].tick_params(axis='both', which='major', labelsize=12)
    axes[i].tick_params(axis='both', which='minor', labelsize=12)

axes[0].set_xlim((0.9,2.6))
axes[1].set_xlim((0.9,2.6))
axes[2].set_xlim((0.9,2.6))
axes[3].set_xlim((0.9,2.6))
axes[4].set_xlim((0.9,2.6))    
axes[0].set_ylabel('Permuted Feature',size=14)

plt.tight_layout()
plt.show()

#%%
# get the total feature importance for each feature
result = pd.concat(list(results.values()), axis=1)
result_ = result.drop(columns=['feature'])
result_ = result_.mean(axis=1)
result = result.loc[:, ~result.columns.duplicated()]
result['total_rmse'] = result_
result = result.sort_values(by='total_rmse',ascending=True)
print(result_)
print(result)

baseline = result[result.feature=='BASELINE']['total_rmse'].values[0]
print(baseline)
colors = colors[::-1]
fix, axes = plt.subplots(1,1,figsize=(6,6))
axes.barh(np.arange(len(COLS)+1),result.total_rmse.values, color=colors)
axes.set_yticks(np.arange(len(COLS)+1),result.feature.values, size=12)
axes.set_title(f'Total Feature Importance',size=16)
axes.plot([baseline,baseline],
             [-1,len(COLS)+1], '--', color='red', label=f'Baseline \nRMSE={baseline:.3f}')
axes.set_ylim((-1,len(COLS)+1))
axes.set_xlabel(f'RMSE',size=14)
axes.tick_params(axis='both', which='major', labelsize=12)
axes.tick_params(axis='both', which='minor', labelsize=12)
axes.set_xlim((1,2.6))
axes.legend(fontsize=11)
plt.tight_layout()
plt.show()

#%%
from matplotlib.colors import ListedColormap
cm = plt.cm.get_cmap('viridis')
my_cmap = cm(np.linspace(0,1,15))

# Create new colormap
my_cmap = ListedColormap(my_cmap)
#plot a time series for each feature group
groups = np.empty((5,8))

for k in range(len(COLS)):
    groups[0,k]=results_1_[k]
    groups[1,k]=results_2_[k]
    groups[2,k]=results_3_[k]
    groups[3,k]=results_4_[k]
    groups[4,k]=results_5_[k]
    
plt.figure(figsize=(6,4))
plt.plot(groups[:,0],label='Group 1', color=my_cmap(2))
plt.plot(groups[:,1],label='Group 2', color=my_cmap(4))
plt.plot(groups[:,2],label='Group 3', color=my_cmap(5))
plt.plot(groups[:,3],label='Group 4', color=my_cmap(6))
plt.plot(groups[:,4],label='Group 5', color=my_cmap(8))
plt.plot(groups[:,5],label='Group 6', color=my_cmap(10))
plt.plot(groups[:,6],label='Group 7', color=my_cmap(12))
plt.plot(groups[:,7],label='Group 8', color=my_cmap(14))
plt.plot(baselines,label='Baseline', color='red', linestyle='--')
plt.ylabel('RMSE')
plt.xticks(np.arange(5),['Day 1','Day 2','Day 3','Day 4','Day 5'])
#plt.ylim((0.5,2.6))
plt.legend()
plt.show()

#%%


gtl_COLS = ['Chlorophyll-a','Cloudmask', 'Sea surface temperature', 'Sine of day of year', 'Air temperature',
            'Radiation', 'Cosine of day of year', 'Precipitation', 'Wind v10',  'Wave VHM0', 'Wind u10', 'Topography', 'Longitude', 'Wave VMDR', 'Latitude']
riga_COLS = ['Chlorophyll-a','Cloudmask', 'Latitude', 'Radiation', 'Sea surface temperature', 'Cosine of day of year', 'Air temperature',
            'Sine of day of year', 'Wind v10', 'Precipitation', 'Wave VHM0', 'Topography', 'Longitude', 'Wave VMDR', 'Wind u10']
ktt_COLS = ['Chlorophyll-a','Cloudmask', 'Topography', 'Radiation', 'Sea surface temperature', 'Cosine of day of year', 'Air temperature',
            'Sine of day of year', 'Wind v10', 'Precipitation', 'Wave VHM0', 'Latitude', 'Longitude', 'Wave VMDR', 'Wind u10']


riga_baselines = [3.0465, 3.2528, 3.4136, 3.6419, 3.6796]
riga_results_1_ = [3.7099, 3.7099, 3.7099, 3.1150, 3.1150, 3.1150, 3.1150, 3.0594, 3.2268, 3.0657, 3.0657, 3.0598, 3.2862, 3.0020, 3.0020]
riga_results_2_ = [3.7248, 3.7248, 3.7248, 3.4463, 3.4463, 3.4463, 3.4463, 3.2309, 3.3828, 3.3931, 3.3931, 3.2943, 3.4360, 3.2065, 3.2065]
riga_results_3_ = [3.7219, 3.7219, 3.7219, 3.4691, 3.4691, 3.4691, 3.4691, 3.3636, 3.3919, 3.5518, 3.5518, 3.4612, 3.5842, 3.3792, 3.3792]
riga_results_4_ = [3.8258, 3.8258, 3.8258, 3.7196, 3.7196, 3.7196, 3.7196, 3.6076, 3.6266, 3.7122, 3.7122, 3.6733, 3.7548, 3.6387, 3.6387]
riga_results_5_ = [3.7556, 3.7556, 3.7556, 3.7518, 3.7518, 3.7518, 3.7518, 3.5998, 3.6124, 3.7724, 3.7724, 3.7126, 3.8197, 3.6951, 3.6951]

gtl_baselines = [1.1196, 1.0241, 1.0892, 1.1814, 1.2819]
gtl_results_1_ = [1.5839, 1.5839, 1.2520, 1.2520, 1.2520, 1.0941, 1.0941, 1.0941, 1.0940, 1.1799, 1.1799, 1.0776, 1.0776, 1.3803, 1.0859]
gtl_results_2_ = [1.5698, 1.5698, 1.3144, 1.3144, 1.3144, 1.1205, 1.1205, 1.1205, 1.1732, 1.2413, 1.2413, 0.9725, 0.9725, 1.3928, 0.9929]
gtl_results_3_ = [1.6318, 1.6318, 1.4326, 1.4326, 1.4326, 1.1945, 1.1945, 1.1945, 1.2059, 1.3121, 1.3121, 1.0653, 1.0653, 1.2989, 1.0825]
gtl_results_4_ = [1.5342, 1.5342, 1.3679, 1.3679, 1.3679, 1.2338, 1.2338, 1.2338, 1.1504, 1.2111, 1.2111, 1.1169, 1.1169, 1.2856, 1.1151]
gtl_results_5_ = [1.7150, 1.7150, 1.4016, 1.4016, 1.4016, 1.3491, 1.3491, 1.3491, 1.2927, 1.3454, 1.3454, 1.2431, 1.2431, 1.4530, 1.2498]

ktt_baselines = [1.4431, 1.4536, 1.4671, 1.4644, 1.4670]
ktt_results_1_ = [2.4749, 2.4749, 2.4749, 1.3004, 1.3004, 1.3004, 1.3004, 1.3665, 1.3593, 1.5293, 1.5293, 1.3292, 1.3382, 1.3390, 1.3390]
ktt_results_2_ = [2.4800, 2.4800, 2.4800, 1.3519, 1.3519, 1.3519, 1.3519, 1.3804, 1.4012, 1.4721, 1.4721, 1.3746, 1.3632, 1.3005, 1.3005]
ktt_results_3_ = [2.5249, 2.5249, 2.5249, 1.4185, 1.4185, 1.4185, 1.4185, 1.4347, 1.4684, 1.5398, 1.5398, 1.4264, 1.4221, 1.3623, 1.3623]
ktt_results_4_ = [2.5186, 2.5186, 2.5186, 1.4082, 1.4082, 1.4082, 1.4082, 1.4259, 1.4104, 1.5315, 1.5315, 1.4384, 1.3724, 1.3385, 1.3385]
ktt_results_5_ = [2.5041, 2.5041, 2.5041, 1.4627, 1.4627, 1.4627, 1.4627, 1.4744, 1.4590, 1.5577, 1.5577, 1.4929, 1.4077, 1.4219, 1.4219]

ktt_stacked = np.vstack((ktt_results_1_,ktt_results_2_,ktt_results_3_,ktt_results_4_,ktt_results_5_))
gtl_stacked = np.vstack((gtl_results_1_,gtl_results_2_,gtl_results_3_,gtl_results_4_,gtl_results_5_))
riga_stacked = np.vstack((riga_results_1_,riga_results_2_,riga_results_3_,riga_results_4_,riga_results_5_))

df_ktt = pd.DataFrame(columns=ktt_COLS)
df_riga = pd.DataFrame(columns=riga_COLS)
df_gtl = pd.DataFrame(columns=gtl_COLS)

for c in range(len(ktt_COLS)):
    df_ktt[ktt_COLS[c]] = [np.mean(ktt_stacked[:,c])]
    
for c in range(len(riga_COLS)):
    df_riga[riga_COLS[c]] = [np.mean(riga_stacked[:,c])]
for c in range(len(gtl_COLS)):
    df_gtl[gtl_COLS[c]] = [np.mean(gtl_stacked[:,c])]

# concat the three datframses on the same column
df = pd.concat([df_ktt, df_riga, df_gtl], axis=0).reset_index(drop=True)  
df

#%%
# make a vertical barchart, where every columns is in one location with dofferent colours for the different rows
# each columns is a different bar
# each row is a different colour

# Colors for each row
colors = ['red', 'green', 'blue']

# Transpose the DataFrame
df_transposed = df.T

# Create the bar plot
plt.figure(figsize=(20, 10))  # Set the figure size (width, height)
plt.barh(df_transposed.index, df_transposed, color=colors[0], label='Southern Kattegat', alpha=0.5)
# plt.barh(df_transposed.index, df_transposed.iloc[:, 1], color=colors[1], left=df_transposed.iloc[:, 0], label='Gulf of Riga', alpha=0.5)
# plt.barh(df_transposed.index, df_transposed.iloc[:, 2], color=colors[2], left=df_transposed.iloc[:, 0] + df_transposed.iloc[:, 1], label='Gotland Basin', alpha=0.5)
plt.axvline(0, color='black', linestyle='--')
# Set labels and title
plt.xlabel('Deviation from Baseline', fontsize=20)

plt.title('Horizontal Bar Plot', fontsize=30)

# Add a legend
plt.legend(fontsize=20)

# Display the plot
plt.show()

#%%
# Create figure and axis objects
fig, ax = plt.subplots()

# Set the colors for each row
colors = ['red', 'green', 'blue']

# Plot horizontal bars for each row and label them
for i, row in enumerate(df.iterrows()):
    row_values = row[1].values
    bar_positions = np.arange(len(row_values))
    bar_labels = df.columns
    ax.barh(bar_positions, row_values, align='center', color=colors[i], label=f'Row {i+1}', alpha=0.5)
    
# Add a vertical line at 0.0
ax.axvline(x=0.0, color='black')

# Set labels and title
ax.set_yticks(np.arange(len(df.columns)))
ax.set_yticklabels(df.columns)
ax.set_xlabel('Value')
ax.set_ylabel('Columns')
ax.set_title('Bar Plot')

# Add legend
ax.legend()

# Show the plot
plt.show()

#%%
# Create figure and axis objects
fig, ax = plt.subplots(figsize=(20, 15))

# Set the colors for each row
colors = ['red', 'orange', 'blue']
labels = ['Southern Kattegat', 'Gulf of Riga', 'Gotland Basin']

# Plot lollipops for each row and label them
for i, row in enumerate(df.iterrows()):
    row_values = row[1].values
    bar_positions = np.arange(len(row_values))
    bar_labels = df.columns
    ax.hlines(bar_positions, [0], row_values, color=colors[i], label=labels[i], linewidth=3, alpha=0.5)

for i, row in enumerate(df.iterrows()):
    row_values = row[1].values
    bar_positions = np.arange(len(row_values))
    # increase the marker size
    ax.plot(row_values, bar_positions, marker='o', linestyle='', color=colors[i], markersize=15)

# Add a vertical line at 0.0
ax.axvline(x=0.0, color='black')

# Set labels and title
ax.set_yticks(np.arange(len(df.columns)), fontsize=25)
ax.set_yticklabels(df.columns, fontsize=25)
plt.xlabel('Deviation from Baseline', fontsize=25)

# increase fotsize of xticks
plt.xticks(fontsize=20)

# Add legend
ax.legend(fontsize=25)

# Show the plot
plt.show()


#%%
# create a horizontal barplot with the different columns as different bars
# each row is a different colour


    


# %%
# visualize cosine of day of year
days = np.arange(365)
days = np.cos(2 * np.pi * days/365.25)

plt.figure(figsize=(6,4))
plt.plot(days)
plt.show()



# %%
