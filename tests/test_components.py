import torch

class TestLlamaComponents:
    # pytest -q -s tests/test_components.py
    def test_rotary(self):
        dim = 8
        theta = 10000
        max_seq_len = 16
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(max_seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        print("freqs_cis.shape = ", freqs_cis.shape)
        print(freqs_cis)

        bs = 10
        seqlen = 7
        nh = 16
        nh_kv = nh // 8
        hd = dim
        start_pos = 3
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        xq = torch.randn([bs, seqlen, nh, hd], dtype=torch.float)
        xk = torch.randn([bs, seqlen, nh_kv, hd], dtype=torch.float)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
            ndim = x.ndim
            assert 0 <= 1 < ndim
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
