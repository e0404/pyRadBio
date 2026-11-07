# Parameter & unit conventions

| Quantity | Symbol | Unit |
| --- | --- | --- |
| kinetic energy / nucleon | E | MeV/u |
| LET (unrestricted) | LET_inf | keV/um |
| absorbed dose / specific energy | D, z | Gy |
| linear LQ coefficient | alpha | Gy^-1 |
| quadratic LQ coefficient | beta | Gy^-2 |
| nucleus / domain radius | R_n, r_d | um |
| track radii | r_min, r_max | nm (params) / um (computation) |

Mixing
- alpha (and z*): dose-average  ->  sum(d_i x_i)/sum(d_i)
- beta: dose-averaged sqrt  ->  (sum d_i sqrt(b_i)/sum d_i)^2  (Zaider-Rossi/Kanai)

Spectral integration
- Integrate stopping_power(E) * quantity(E) on a ~16-node log sub-grid per bin.
  Never at bin centres (intra-bin S x quantity covariance is significant).

Photon survival (LEM)
- Shouldered LQ for D<=D_t; exponential tail above with s_max=alpha_x+2 beta_x D_t.
- LEM II/IV: high-dose tail enhanced by eta(D) (clustering).

Radial dose (LEM)
- r_max = gamma E^delta; gamma=0.062 um (MeV/u)^-1.7, delta=1.7 (LEM II/III).
  (LEM I original used gamma=0.05 — keep version-specific.)
- r_min: LEM I ~10 nm; LEM II 0.3 nm + sigma=4 nm radical diffusion;
  LEM III r_min(beta)=beta*r_c, r_c=40 nm.
