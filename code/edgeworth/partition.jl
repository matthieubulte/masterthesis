function _partitions(target, nterms, currentseq)::Array{Array{Int, 1}}
  # overshoot or undershoot
  if target < 0 || (target > 0 && nterms == 0)
    return []
  end

  # reached target
  if nterms == 0
    return [currentseq]
  end

  # otherwise, check other possible terms
  workingseqs = []
  for j = 1:length(currentseq)
    newtarget = target - j   
    newseq = copy(currentseq)
    newseq[j] += 1

    workingseqs_forj = _partitions(newtarget, nterms-1, newseq)
    workingseqs = vcat(workingseqs, workingseqs_forj)
  end

  return workingseqs
end


# This function computes the partition sets (r1, ..., rl) such
# that r1 * 1 + r2 * 2 + ... + rl * l = target
function partitions(target)
  maxlen::Int = target
  workingseqs = []
  emptyseq = repeat([0], maxlen)

  for len=1:maxlen
    workingseqs_for_len = _partitions(target, len, emptyseq)
    workingseqs = vcat(workingseqs, workingseqs_for_len)
  end

  return unique(workingseqs)
end

###### Implementation of Edgeworth expansion with hand made coefficients

# function P(n, lambdas)
#   x = Polynomial([0, 1])
#   res = 0

#   parts = partitions(n)

#   for part = parts
#     part_val = 1
#     for j = 1:length(part)
#       occ = part[j]
#       pid = j+2
#       part_val *= (lambdas[pid] / factorial(pid))^occ / factorial(occ) * x^(pid*occ)
#     end
#     res += part_val
#   end

#   return res
# end

# function H(n, x)
#   # return basis(ChebyshevHermite, n)(x)
#   return symbols("H$n")
# end

# function edgeworth(fam, order, n, x)
#   res = 0
#   t = Taylor1(typeof(x), order*2)

#   scaled_cumulants = cumulant_gen_fn(fam, t).coeffs
#   scaling = exp(t).coeffs
#   cumulants = scaled_cumulants ./ scaling
  
#   for j = 1:order
#     p = P(j, cumulants[2:end])
#     res += sum([ 
#       p.coeffs[k+1] * H(k, x)
#       for k=3:length(p.coeffs)-1
#     ]) / (n^(j/2))
#   end

#   return 1 + res
# end

