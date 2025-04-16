def condition_score2(self, cond_fn, p_mean_var, x, t, model_kwargs=None, classifier=None, 
                     t_set=[], cond_fn2=None):
    """
    Compute what the p_mean_variance output would have been, should the
    model's score function be conditioned by cond_fn.
    See condition_mean() for details on cond_fn.
    Unlike condition_mean(), this instead uses the conditioning strategy
    from Song et al (2020).
    """
    t = t.long()
    alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
    eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])

    # Nếu không có classifier hoặc t[0] không nằm trong t_set thì sử dụng cfn ban đầu
    if (classifier is None) or (t[0] not in t_set):
        a, cfn = cond_fn2(x, self._scale_timesteps(t).long(), **model_kwargs)
        eps = eps - (1 - alpha_bar).sqrt() * cfn

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out, cfn  # cfn chính là saliency

    else:
        out = p_mean_var.copy()
        a, cfn = cond_fn2(x, self._scale_timesteps(t).long(), **model_kwargs)

        # Tạo bản sao của cfn để tối ưu
        cfn_optim = cfn.detach().clone().requires_grad_(True)
        print('cfn_optim grad: ', cfn_optim.requires_grad)
        optimizer = th.optim.AdamW([cfn_optim], lr=0.001)
        lambda_eff = 0.1  # Hệ số cân bằng giữa việc giữ logits và phạt regularization

        with th.enable_grad():
            for i in range(20):
                optimizer.zero_grad()

                # Tính mean mới với cfn_optim được cập nhật
                mean_new = out["mean"] + out["variance"] * cfn_optim
                print('mean_new grad: ', mean_new.requires_grad)
                logits_new = classifier(mean_new, timesteps=t-1)

                # Hàm mất mát: giữ logits không thay đổi và regularization L1 cho cfn_optim
                loss_logits = F.cross_entropy(logits_new, model_kwargs['y'], reduction="none")
                loss_logits = loss_logits.mean()
                loss_reg = th.mean(th.abs(cfn_optim))
                loss = loss_logits + lambda_eff * loss_reg

                # Tính gradient riêng cho từng thành phần loss
                grad_loss_logits = th.autograd.grad(loss_logits, cfn_optim, retain_graph=True)[0]
                grad_loss_reg = th.autograd.grad(loss_reg, cfn_optim, retain_graph=True)[0]

                # In ra gradient của từng thành phần
                print(f"[Iter {i}] Grad của loss_logits: {grad_loss_logits.detach()}")
                print(f"[Iter {i}] Grad của loss_reg: {grad_loss_reg.detach()}")

                # Sau đó thực hiện backward tổng hợp trên loss
                loss.backward()
                print(f"[Iter {i}] Tổng Grad (sau backward): {cfn_optim.grad.detach()}")

                optimizer.step()

        # Sau tối ưu, cập nhật cfn với giá trị của cfn_optim
        cfn_updated = cfn_optim.detach()

        # Cập nhật final eps dựa trên cfn đã được điều chỉnh
        eps = eps - (1 - alpha_bar).sqrt() * cfn_updated
        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out, cfn_updated
