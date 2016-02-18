import modules
import nn


class TrackerSet(nn.ParametrizedBlock):
  def __init__(self, input_h_size, input_s_size, n_trackers):
    self.trackers = []
    for i in range(n_trackers):
      self.trackers.append(modules.Tracker(input_h_size, input_s_size))

    self.parametrize_from_layers(
      self.trackers,
      ["tracker%d" % i for i in range(len(self.trackers))]
    )

  def forward(self, inputs):
    assert type(inputs) == tuple

    h_t = inputs[0]
    s = inputs[1]
    input_tr_nlu = inputs[2:]
    tracker_input_tr = input_tr_nlu[:len(input_tr_nlu) / 2]
    tracker_input_slu = input_tr_nlu[len(input_tr_nlu) / 2:]

    assert len(tracker_input_tr) == len(tracker_input_slu)
    assert len(tracker_input_tr) == len(self.trackers)

    res = []
    res_aux = []
    for tracker, tr, slu in zip(self.trackers, tracker_input_tr, tracker_input_slu):
      ((tracker_output,), tracker_output_aux) = tracker.forward((tr, slu, h_t, s,))
      res.append(tracker_output)
      res_aux.append(tracker_output_aux)

    return (tuple(res), nn.Vars(
        res=res_aux
    ))

  def backward(self, aux, dres):
    lst_dtr = []
    lst_dslu = []
    lst_dh_t = []
    lst_ds = []
    for dtracker_output, tracker_output_aux, tracker in zip(dres, aux['res'], self.trackers):
      self.accum_grads((lst_dtr, lst_dslu, lst_dh_t, lst_ds,),
                       tracker.backward(tracker_output_aux, (dtracker_output,)))

    return (sum(lst_dh_t), sum(lst_ds),) + tuple(lst_dtr) + tuple(lst_dslu)
