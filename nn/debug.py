class DEBUG:
  vocab = None
  nlg_external_input = None
  db_entry_dist = None

  @classmethod
  def new_dialog(cls, dialog, labels):
    cls.dialog = dialog
    cls.labels = labels
    cls.nlg_external_input = []
    cls.db_entry_dist = []

  @classmethod
  def add_nlg_external_input(cls, external_input):
    cls.nlg_external_input.append(external_input)

  @classmethod
  def get_nlg_external_input(cls):
    return cls.nlg_external_input

  @classmethod
  def add_db_entry_dist(cls, dist):
    cls.db_entry_dist.append(dist)

  @classmethod
  def get_db_entry_dist(cls):
    return cls.db_entry_dist