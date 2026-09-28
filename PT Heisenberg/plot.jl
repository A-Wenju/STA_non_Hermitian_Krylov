# plot.jl -- draw heisenberg_agp_cascade.pdf from data_cascade.jl's CSV cache.
# Plotting does no simulation, so styling can be iterated on cheaply.
#
# Exact ‖A_ξ‖² (solid black) plus a log-spaced cascade of odd-K truncations,
# M = ⌊K/2⌋ = 1, 2, 4, 8, 16, 32, 64, 124, 249 (dashed, viridis by rank),
# with a hand-drawn colorbar keyed to M.
#
# Usage:
#     julia plot.jl        -> heisenberg_agp_cascade.pdf, heisenberg_agp_cascade.png

using DelimitedFiles, Plots, LaTeXStrings
gr(fontfamily = "Computer Modern")

# ── Input ──────────────────────────────────────────────────────────────────────
CSV_FILE = "heisenberg_agp_cascade.csv"
OUTFILE  = "heisenberg_agp_cascade"
K_VALUES = [3, 5, 9, 17, 33, 65, 129, 249, 499]   # odd K only (even K diverges)

raw, header = readdlm(CSV_FILE, ',', header = true)
header = vec(String.(header))
col(name) = raw[:, findfirst(==(name), header)]

xi    = Float64.(col("xi"))
exact = Float64.(col("exact_norm_sq"))

println("Loaded    : $CSV_FILE  ($(length(xi)) xi points)")
println("K values  : $K_VALUES")
println("M = K÷2   : $(K_VALUES .÷ 2)")

# ── Colormap: viridis, indexed 0..1 by rank across K_VALUES (log-spaced, so
#    rank-indexing spreads colors evenly rather than clustering at low M) ──────
cmap   = cgrad(:viridis, rev = true)
n_K    = length(K_VALUES)
colors = [cmap[(i - 1) / (n_K - 1)] for i in 1:n_K]

# ── Base figure ────────────────────────────────────────────────────────────────
p = plot(
    xlabel         = L"$\xi$",
    ylabel         = L"$‖A_\xi‖^2$",
    yscale         = :log10,
    legend         = :topleft,
    frame          = :box,
    grid           = false,
    fontfamily     = "Computer Modern",
    guidefontsize  = 16,
    tickfontsize   = 14,
    legendfontsize = 12,
    margin         = 3Plots.mm,
)

# one truncated-AGP-norm curve per Krylov depth K (M labels go on the colorbar)
M_VALUES = K_VALUES .÷ 2
for (i, K) in enumerate(K_VALUES)
    plot!(p, xi, Float64.(col("K$K"));
        label     = false,
        color     = colors[i],
        linestyle = :dash,
        lw        = 1.6,
    )
end

# exact reference -- solid black, on top
plot!(p, xi, exact;
    label = "Exact",
    color = :black,
    lw    = 2.2,
)

# 6 colorbar tick values log-spaced across [M_VALUES[1], M_VALUES[end]], placed
# by interpolating linearly in log10(M) between the bracketing rank positions.
function rank_pos_log(target, Ms)
    n = length(Ms)
    logMs = log10.(Ms)
    logt  = log10(target)
    logt <= logMs[1]   && return 1.0
    logt >= logMs[end] && return float(n)
    k = findlast(<=(logt), logMs)
    frac = (logt - logMs[k]) / (logMs[k + 1] - logMs[k])
    return k + frac
end

tick_M      = round.(Int, 10 .^ range(log10(M_VALUES[1]), log10(M_VALUES[end]), length = 6))
tick_idx    = rank_pos_log.(tick_M, Ref(M_VALUES))
tick_labels = string.(tick_M)

# ── Colorbar keyed to M ────────────────────────────────────────────────────────
# Plots.jl's GR backend ignores `colorbar_ticks` (JuliaPlots/Plots.jl#3560), so
# the colorbar is drawn by hand as a narrow high-resolution heatmap strip in its
# own subplot, where `yticks` are respected.
n_fine        = 256
strip_ranks   = range(0.0, 1.0, length = n_fine)
tick_idx_fine = 1 .+ (tick_idx .- 1) .* (n_fine - 1) / (n_K - 1)
p_cbar = heatmap(reshape(strip_ranks, n_fine, 1);
    color         = cmap,
    colorbar      = false,
    xticks        = false,
    yticks        = (tick_idx_fine, tick_labels),
    ymirror       = true,
    framestyle    = :box,
    title         = "M",
    titlefontsize = 12,
    tickfontsize  = 12,
    fontfamily    = "Computer Modern",
)

p_full = plot(p, p_cbar, layout = @layout([a{0.88w} b{0.06w}]), size = (700, 450))

# ── Save ───────────────────────────────────────────────────────────────────────
savefig(p_full, "$OUTFILE.pdf")
savefig(p_full, "$OUTFILE.png")
println("Saved     : $OUTFILE.pdf, $OUTFILE.png")
