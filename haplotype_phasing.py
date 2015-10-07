import numpy as np
import timeit
from copy import deepcopy
from itertools import product

def simulate_haplotype_pool(n_haplotypes, n_snps):

    pool = np.array([[-1 for i in range(n_snps)] for i in range(n_haplotypes)])
    if n_haplotypes > 2 ** n_snps:
        return pool
    for i in range(n_haplotypes):
        haplotype_i = np.random.random_integers(0, 1, n_snps)
        while row_in_list(haplotype_i, pool):
            haplotype_i = np.random.random_integers(0, 1, n_snps)
        pool[i] = haplotype_i
    return pool

def simulate_haplotypes(n_individuals, pool):
    haplotypes = np.zeros((n_individuals * 2, pool.shape[1]), dtype = int)
    for i in range(n_individuals):
        haplotypes[2 * i] = pool[np.random.random_integers(0, pool.shape[0] - 1)]
        haplotypes[2 * i + 1] = pool[np.random.random_integers(0, pool.shape[0] - 1)]
    return haplotypes

def simulate_genotypes(haplotypes):
    genotypes = np.zeros((haplotypes.shape[0] / 2, haplotypes.shape[1]), dtype = int)
    for i in range(int(haplotypes.shape[0] / 2)):
        for j in range(haplotypes.shape[1]):
            genotypes[i, j] = assemble_genotype(haplotypes[2 * i, j], haplotypes[2 * i + 1, j])
    return genotypes

def row_in_list(row, list):
    if len(list) == 0:
        return False
    if len(list.shape) == 1:
        if np.all(list == row):
            return True
    for i in range(list.shape[0]):
        if np.all(list[i] == row):
            return True
    return False

def assemble_genotype(haplotype_1, haplotype_2):
    if haplotype_1 == 0 and haplotype_2 == 0:
        return 0
    elif haplotype_1 == 1 and haplotype_2 == 1:
        return 1
    return 2

def clarks(genotypes):
    haplotypes = np.array([[-1 for i in range(genotypes.shape[1])] for i in range(2 * genotypes.shape[0])])
    haplotype_pool = np.array([], dtype = int)
    remaining = deepcopy(genotypes)
    can_start = False
    #Check for unambiguous
    for i in range(remaining.shape[0]):
        if not ambiguous(remaining[i]):
            haplotypes[(2 * i):(2 * i + 2)] = phase_unambiguous(remaining[i])
            haplotype_1 = haplotypes[2 * i]
            haplotype_2 = haplotypes[2 * i + 1]
            if haplotype_pool.shape[0] == 0:
                haplotype_pool = np.append(haplotype_pool, haplotype_1)
            elif not row_in_list(haplotype_1, haplotype_pool):
                haplotype_pool = np.vstack((haplotype_pool, haplotype_1))
            if not row_in_list(haplotype_2, haplotype_pool):
                haplotype_pool = np.vstack((haplotype_pool, haplotype_2))
            remaining[i, 0] = -1
            can_start = True
    if not can_start:
        return haplotypes
    while not resolved(remaining):
        updated = False
        for i in range(remaining.shape[0]):
            if remaining[i, 0] != -1:
                matches = matches_in_pool(remaining[i], haplotype_pool)
                if matches[0] != -1 and matches[1] != -1:
                    haplotypes[2 * i] = haplotype_pool[matches[0]]
                    haplotypes[2 * i + 1] = haplotype_pool[matches[1]]
                    remaining[i, 0] = -1
                    updated = True
                elif matches[0] != -1:
                    complement = complementary_haplotype(remaining[i], haplotype_pool[matches[0]])
                    haplotypes[2 * i] = haplotype_pool[matches[0]]
                    haplotypes[2 * i + 1] = complement
                    haplotype_pool = np.vstack((haplotype_pool, complement))
                    remaining[i, 0] = -1
                    updated = True
        if not updated:
            break
    return haplotypes

def ambiguous(genotype):
    num_het = sum(genotype == 2)
    if num_het > 1:
        return True
    return False

def phase_unambiguous(genotype):
    haplotypes = np.zeros((2, len(genotype)), dtype = int)
    for i in range(len(genotype)):
        if genotype[i] == 0:
            haplotypes[0:2, i] = [0, 0]
        elif genotype[i] == 1:
            haplotypes[0:2, i] = [1, 1]
        elif genotype[i] == 2:
            haplotypes[0:2, i] = [0, 1]
    return haplotypes

def resolved(genotypes):
    for i in range(len(genotypes)):
        if genotypes[i, 0] != -1:
            return False
    return True

def complementary_haplotype(genotype, haplotype):
    complement = np.array([-1 for i in range(len(haplotype))])
    for i in range(len(genotype)):
        if genotype[i] == 0 and haplotype[i] == 0:
            complement[i] = 0
        elif genotype[i] == 1 and haplotype[i] == 1:
            complement[i] = 1
        elif genotype[i] == 2:
            if haplotype[i] == 0:
                complement[i] = 1
            elif haplotype[i] == 1:
                complement[i] = 0
    return complement

def valid_haplotype(genotype, haplotype):
    return np.all(complementary_haplotype(genotype, haplotype) != -1)
        
def matches_in_pool(genotype, pool):
    matches = [-1, -1]
    if len(pool) == 0:
        return matches
    if len(pool.shape) == 1:
        if np.all(genotype == pool):
            matches[0] = 0
        return matches
    for i in range(pool.shape[0]):
        if valid_haplotype(genotype, pool[i]):
            matches[0] = i
            complement = complementary_haplotype(genotype, pool[i])
            for j in range(pool.shape[0]):
                if np.all(pool[j] == complement):
                    matches[1] = j
        if matches[0] != -1 and matches[1] != -1:
            break
    return matches

def greedy(genotypes):
    haplotypes = np.array([[-1 for i in range(genotypes.shape[1])] for i in range(2 * genotypes.shape[0])])
    haplotype_pool = set()
    phased = [0 for i in range(genotypes.shape[0])]
    while not all(i == 2 for i in phased):
        if sum([i == 2 for i in phased]) == len(phased) - 1:
            index = np.where(np.array(phased) == 1)[0][0]
            complement = complementary_haplotype(genotypes[index], haplotypes[2 * index])
            haplotypes[2 * index + 1] = complement
            haplotype_pool.add(tuple(complement))
            phased[index] = 2
            return haplotypes
        else:
            max = 0
            index_1 = []
            index_2 = []
            current_pair = ()
            perms = set([i for i in product((0, 1), repeat = genotypes.shape[1])]) - haplotype_pool
            pairs = [i for i in product(perms, perms)]
            for pair in pairs:
                count = 0
                current_index_1 = []
                current_index_2 = []
                for i in range(genotypes.shape[0]):
                    if phased[i] == 0:
                        if valid_haplotype(genotypes[i], pair[0]):
                            count += 1
                            complement = complementary_haplotype(genotypes[i], pair[0])
                            if np.all(complement == pair[0]) or np.all(complement == pair[1]):
                                count += 1
                                current_index_2.append(i)
                            else:
                                current_index_1.append(i)
                        elif valid_haplotype(genotypes[i], pair[1]):
                            count += 1
                            complement = complementary_haplotype(genotypes[i], pair[1])
                            if np.all(complement == pair[0]) or np.all(complement == pair[1]):
                                count += 1
                                current_index_2.append(i)
                            else:
                                current_index_1.append(i)
                    elif phased[i] == 1:
                        complement = complementary_haplotype(genotypes[i], haplotypes[2 * i])
                        if np.all(complement == pair[0]) or np.all(complement == pair[1]):
                            count += 1
                            current_index_1.append(i)
                if count > max:
                    max = count
                    index_1 = current_index_1
                    index_2 = current_index_2
                    current_pair = pair
            for i in index_1:
                if phased[i] == 0:
                    if valid_haplotype(genotypes[i], current_pair[0]):
                        haplotypes[2 * i] = current_pair[0]
                        haplotype_pool.add(current_pair[0])
                    elif valid_haplotype(genotypes[i], current_pair[1]):
                        haplotypes[2 * i] = current_pair[1]
                        haplotype_pool.add(current_pair[1])
                    phased[i] = 1
                elif phased[i] == 1:
                    complement = complementary_haplotype(genotypes[i], haplotypes[2 * i])
                    if np.all(complement == current_pair[0]):
                        haplotypes[2 * i + 1] = current_pair[0]
                        haplotype_pool.add(current_pair[0])
                    elif np.all(complement == current_pair[1]):
                        haplotypes[2 * i + 1] = current_pair[1]
                        haplotype_pool.add(current_pair[1])
                    phased[i] = 2
            for i in index_2:
                if valid_haplotype(genotypes[i], current_pair[0]):
                    haplotypes[2 * i] = current_pair[0]
                    haplotype_pool.add(current_pair[0])
                    complement = complementary_haplotype(genotypes[i], haplotypes[2 * i])
                    if np.all(complement == current_pair[0]):
                        haplotypes[2 * i + 1] = current_pair[0]
                    elif np.all(complement == current_pair[1]):
                        haplotypes[2 * i + 1] = current_pair[1]
                        haplotype_pool.add(current_pair[1])
                elif valid_haplotype(genotypes[i], current_pair[1]):
                    haplotypes[2 * i] = current_pair[1]
                    haplotype_pool.add(current_pair[1])
                    complement = complementary_haplotype(genotypes[i], haplotypes[2 * i])
                    if np.all(complement == current_pair[1]):
                        haplotypes[2 * i + 1] = current_pair[1]
                    elif np.all(complement == current_pair[0]):
                        haplotypes[2 * i + 1] = current_pair[0]
                        haplotype_pool.add(current_pair[0])
                phased[i] = 2
    return haplotypes

def window_greedy(genotypes, window, overlap):
    increment = window - overlap
    iterations = 1
    count = window
    while count <= genotypes.shape[1]:
        count += increment
        iterations += 1
    index = 0
    hap_windows = []
    for i in range(iterations):
        gen_window = genotypes[:, index:index + window]
        hap_windows.append(greedy(gen_window))
        index += increment
    haplotypes = []
    for hap in hap_windows:
        if len(haplotypes) == 0:
            haplotypes = hap
        else:
            haplotypes = merge(haplotypes, hap, window, overlap)
    return haplotypes

def window_clarks(genotypes, window, overlap):
    increment = window - overlap
    iterations = 1
    count = window
    while count <= genotypes.shape[1]:
        count += increment
        iterations += 1
    index = 0
    hap_windows = []
    for i in range(iterations):
        gen_window = genotypes[:, index:index + window]
        hap_windows.append(clarks(gen_window))
        index += increment
    haplotypes = []
    for hap in hap_windows:
        if len(haplotypes) == 0:
            haplotypes = hap
        else:
            haplotypes = merge(haplotypes, hap, window, overlap)
    return haplotypes

def merge(haps_1, haps_2, window, overlap):
    n_rows = haps_1.shape[0]
    n_columns = haps_1.shape[1] + haps_2.shape[1] - overlap
    to_merge = haps_2[:, overlap:window]
    if len(to_merge.shape) == 1:
        to_merge = to_merge.reshape(n_rows, 1)
    merged = np.zeros((n_rows, n_columns), dtype = int)
    for i in range(int(n_rows/2)):
        if haps_1[2 * i, haps_1.shape[1] - 2] == haps_2[2 * i, 0] and haps_1[2 * i, haps_1.shape[1] - 1] == haps_2[2 * i, 1]:
            merged[2 * i] = np.hstack((haps_1[2 * i], to_merge[2 * i]))
            merged[2 * i + 1] = np.hstack((haps_1[2 * i + 1], to_merge[2 * i + 1]))
        else:
            merged[2 * i] = np.hstack((haps_1[2 * i], to_merge[2 * i + 1]))
            merged[2 * i + 1] = np.hstack((haps_1[2 * i + 1], to_merge[2 * i]))
    return merged

def pool_from_haplotypes(haplotypes):
    pool = set()
    for i in range(haplotypes.shape[0]):
        pool.add(tuple(haplotypes[i]))
    return pool

n_snps = 6
n_pool = 20
n_individuals = 5
iterations = 10

pool = simulate_haplotype_pool(n_pool, n_snps)
haplotypes = simulate_haplotypes(n_individuals, pool)
genotypes = simulate_genotypes(haplotypes)
#sol_wg = window_greedy(genotypes, 5, 2)
print("Original Hapotypes:")
print(haplotypes)
print("Number of original haplotypes = %s" %len(pool_from_haplotypes(haplotypes)))
print("Genotypes:")
print(genotypes)
print("Phased Haplotypes:")
#print(sol_wg)
#print("Number of Haplotypes = %s" %len(pool_from_haplotypes(sol_wg)))
sol = greedy(genotypes)
print(sol)
print("Number of Haplotypes = %s" %len(pool_from_haplotypes(sol)))

'''
#Sliding Window Greedy Benchmarks

average_time_wg = 0
average_accuracy_wg = 0
average_n_haps_wg = 0

for i in range(iterations):
    pool = simulate_haplotype_pool(n_pool, n_snps)
    haplotypes = simulate_haplotypes(n_individuals, pool)
    genotypes = simulate_genotypes(haplotypes)

    start = timeit.default_timer()
    sol_wg = window_greedy(genotypes, 5, 2)
    stop = timeit.default_timer()
    average_time_wg += (stop - start)/iterations
    average_accuracy_wg += (sum(sum(sol_wg == haplotypes))/(sol_wg.shape[0] * sol_wg.shape[1]))/iterations
    average_n_haps_wg += len(pool_from_haplotypes(sol_wg))/iterations

print("Sliding Window Greedy:")
print("Average time = %s seconds" %average_time_wg)
print("Average accuracy = %s" %average_accuracy_wg)
print("Average # haplotypes = %s" %average_n_haps_wg)
'''

'''
#All Benchmarks: Greedy, Sliding Window Greedy, Clark's, Sliding Window Clark's

average_time_g = 0
average_accuracy_g = 0
average_n_haps_g = 0

average_time_wg = 0
average_accuracy_wg = 0
average_n_haps_wg = 0

average_time_c = 0
average_accuracy_c = 0
average_n_haps_c = 0

average_time_wc = 0
average_accuracy_wc = 0
average_n_haps_wc = 0

for i in range(iterations):
    pool = simulate_haplotype_pool(n_pool, n_snps)
    haplotypes = simulate_haplotypes(n_individuals, pool)
    genotypes = simulate_genotypes(haplotypes)

    start = timeit.default_timer()
    sol_g = greedy(genotypes)
    stop = timeit.default_timer()
    average_time_g += (stop - start)/iterations
    average_accuracy_g += (sum(sum(sol_g == haplotypes))/(sol_g.shape[0] * sol_g.shape[1]))/iterations
    average_n_haps_g += len(pool_from_haplotypes(sol_g))/iterations

    start = timeit.default_timer()
    sol_wg = window_greedy(genotypes, 5, 2)
    stop = timeit.default_timer()
    average_time_wg += (stop - start)/iterations
    average_accuracy_wg += (sum(sum(sol_wg == haplotypes))/(sol_wg.shape[0] * sol_wg.shape[1]))/iterations
    average_n_haps_wg += len(pool_from_haplotypes(sol_wg))/iterations

    start = timeit.default_timer()
    sol_c = clarks(genotypes)
    stop = timeit.default_timer()
    average_time_c += (stop - start)/iterations
    average_accuracy_c += (sum(sum(sol_c == haplotypes))/(sol_c.shape[0] * sol_c.shape[1]))/iterations
    average_n_haps_c += len(pool_from_haplotypes(sol_c))/iterations

    start = timeit.default_timer()
    sol_wc = window_clarks(genotypes, 5, 2)
    stop = timeit.default_timer()
    average_time_wc += (stop - start)/iterations
    average_accuracy_wc += (sum(sum(sol_wc == haplotypes))/(sol_wc.shape[0] * sol_wc.shape[1]))/iterations
    average_n_haps_wc += len(pool_from_haplotypes(sol_wc))/iterations

print("Greedy:")
print("Average time = %s seconds" %average_time_g)
print("Average accuracy = %s" %average_accuracy_g)
print("Average # haplotypes = %s" %average_n_haps_g)

print("Sliding Window Greedy:")
print("Average time = %s seconds" %average_time_wg)
print("Average accuracy = %s" %average_accuracy_wg)
print("Average # haplotypes = %s" %average_n_haps_wg)

print("Clark's:")
print("Average time = %s seconds" %average_time_c)
print("Average accuracy = %s" %average_accuracy_c)
print("Average # haplotypes = %s" %average_n_haps_c)

print("Sliding Window Clark's:")
print("Average time = %s seconds" %average_time_wc)
print("Average accuracy = %s" %average_accuracy_wc)
print("Average # haplotypes = %s" %average_n_haps_wc)
'''