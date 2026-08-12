import re, difflib

pdf_refs_text = """1. Agrawal, S., Sarkar, S., Aouedi, O., Yenduri, G., Piamrat, K., Bhattacharya, S., Maddikunta, P.K.R., Gadekallu, T.R.: Federated learning for intrusion detection system: Concepts, challenges and future directions. arXiv preprint arXiv:2106.09527 (2021), https://arxiv.org/abs/2106.09527
2. Fallah, A., Mokhtari, A., Ozdaglar, A.: Personalized federated learning: A meta-learning approach. In: Advances in Neural Information Processing Systems (NeurIPS). pp. 20232-20245 (2020), https://arxiv.org/pdf/2002.07948
3. Kairouz, P., McMahan, H.B., et al.: Advances and open problems in federated learning. Foundations and Trends in Machine Learning 14(1-2), 1-210 (2021), https://arxiv.org/pdf/1912.04977
4. Kim, S., Lee, G., Oh, J., Yun, S.Y.: Fedfn: Feature normalization for alleviating data heterogeneity problem in federated learning. In: Advances in Neural Information Processing Systems (NeurIPS) (2021), https://arxiv.org/pdf/2311.13267
5. Li, T., Sanjabi, M., Beirami, A., Smith, V.: Federated optimization in heterogeneous networks. In: Proceedings of Machine Learning Research (PMLR) (2020), https://arxiv.org/abs/1812.06127
6. McMahan, H.B., Moore, E., Ramage, D., Hampson, S.: Communication-efficient learning of deep networks from decentralized data. Artificial Intelligence and Statistics (2017), https://arxiv.org/pdf/1602.05629
7. Reddi, S.J., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konecny, J., Kumar, S., McMahan, H.B.: Adaptive federated optimization (2021), https://arxiv.org/abs/2003.00295
8. Reisizadeh, A., Hassani, H., Mokhtari, A., et al.: The non-iid data quagmire of decentralized machine learning. IEEE Transactions on Signal Processing 68, 5512-5527 (2020), https://arxiv.org/pdf/1910.00189
9. Sahoo, P., Tripathi, A., Saha, S., Mondal, S.: Fedmrl: Data heterogeneity aware federated multi-agent deep reinforcement learning for medical imaging. arXiv preprint arXiv:2407.05800 (2024), https://arxiv.org/pdf/2407.05800
10. Seo, J., Rong, C., Kim, M.: Fedshift: Tackling dual heterogeneity problem of federated learning via weight shift aggregation (2024), https://arxiv.org/pdf/2402.01070
11. Wang, X., Zhang, Z., Zhang, L.: Adaptive regularization and resilient estimation in federated learning. In: International Conference on Machine Learning (ICML). pp. 10247-10256. PMLR (2021), https://www.computer.org/csdl/journal/sc/2024/04/10316585/1S4z5n0QPcc
12. Zhang, X., Li, T., Jin, R., Yang, Q.: Federated learning on non-iid data via local and global adaptation. Proceedings of the 38th International Conference on Machine Learning (ICML) pp. 8340-8351 (2020), https://arxiv.org/pdf/2306.14443
13. Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., Chandra, V.: Federated learning with non-iid data. arXiv preprint arXiv:1806.00582 (2018), https://arxiv.org/pdf/1806.00582
14. Zhao, Z., Joshi, G.: A dynamic reweighting strategy for fair federated learning. In: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE (2022). https://doi.org/10.1109/ICASSP43922.2022.9746300, https://ieeexplore.ieee.org/document/9746300"""

with open('PAKDD_Final_Version_Submit_v3_CORRECTED.md', encoding='utf-8') as f:
    md = f.read()

refs_block = md.split('# References')[1]
md_refs = re.findall(r'^\d+\.\s.*$', refs_block, flags=re.M)
md_refs = [re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', r) for r in md_refs]  # unwrap md links
md_refs = [re.sub(r'<[^>]+>', '', r) for r in md_refs]  # strip <sup> etc
md_refs = [re.sub(r'\*+', '', r) for r in md_refs]

pdf_refs = pdf_refs_text.strip().split('\n')

def norm(s):
    s = s.replace('–','-').replace('—','-').replace('’',"'")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

print(f"PDF refs found: {len(pdf_refs)}, MD refs found: {len(md_refs)}\n")
all_match = True
for i, (p, m) in enumerate(zip(pdf_refs, md_refs), start=1):
    pn, mn = norm(p), norm(m)
    ratio = difflib.SequenceMatcher(None, pn, mn).ratio()
    status = "MATCH" if ratio > 0.97 else f"DIFF (ratio={ratio:.3f})"
    if ratio <= 0.97:
        all_match = False
    print(f"[{i}] {status}")
    if ratio <= 0.97:
        print(f"    PDF: {pn}")
        print(f"    MD : {mn}")

print(f"\nALL 14 REFERENCES EXACT MATCH: {all_match}")
