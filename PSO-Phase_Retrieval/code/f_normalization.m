function [out] = f_normalization(in)

out = (in - min(min(in)))/(max(max(in)) - min(min(in)));
end