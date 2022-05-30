import numpy as np

def guided_filter(_in, guide, ksize=33, eps=1e-5):
  output = np.zeros_like(_in)

  radius = ksize // 2
  pad_in = np.pad(_in, radius, 'symmetric')
  pad_guide = np.pad(guide, radius, 'symmetric')

  m_guide = np.zeros_like(guide)
  v_guide = np.zeros_like(guide)

  for i in range(guide.shape[0]):
    for j in range(guide.shape[1]):
      win_guide = pad_guide[i:i+ksize, j:j+ksize]
      mk = np.mean(win_guide)
      vk = np.mean((win_guide - mk) ** 2)
      m_guide[i,j] = mk; v_guide[i,j] = vk;

  a = np.zeros_like(_in)
  b = np.zeros_like(_in)

  for i in range(guide.shape[0]):
    for j in range(guide.shape[1]):
      win_input = pad_input[i:i+ksize, j:j+ksize]
      win_guide = pad_guide[i:i+ksize, j:j+ksize]
      m_win_input = np.mean(win_input)

      ak = np.mean(win_guide * win_input - m_guide[i,j] * m_win_input)
      ak = ak / (v_guide[i,j] + eps)
      a[i,j] = ak

      bk = m_win_input - ak * m_guide[i,j]
      b[i,j] = bk

  pad_a = np.pad(a, radius, 'symmetric')
  pad_b = np.pad(b, radius, 'symmetric')

  for i in range(guide.shape[0]):
    for j in range(guide.shape[1]):
      win_ak = pad_a[i:i+ksize, j:j+ksize]
      win_bk = pad_b[i:i+ksize, j:j+ksize]
      output[i,j] = np.mean(win_ak * guide[i,j] + win_bk)

  return output
