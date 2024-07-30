import os

import gspread


class GoogleSSClient:
    def __init__(self) -> None:
        self.gc = gspread.oauth(
            credentials_filename="client_secret.json",
            authorized_user_filename="authorized_user.json"
            )
        
        self.workbook_key = os.environ.get('Spreadsheet_Workbook_Key')

    def write_to_spreadsheet(
            self,
            quant: str,
            scores: list[float],
            non_ja_responses: float,
            infinite_repetitions: float,
            tps_list: list[float],
            usage_vram: float
            ):
        
        workbook = self.gc.open_by_key(self.workbook_key)
        worksheet = workbook.get_worksheet(0)

        quantization_methods = worksheet.col_values(2)
        for i, q in enumerate(quantization_methods):
            if q == quant:
                break
        
        target_row = i + 1

        for n, score in enumerate(scores):
            worksheet.update_cell(target_row, 5 + n, score)
        
        worksheet.update_cell(target_row, 14, non_ja_responses)
        worksheet.update_cell(target_row, 15, infinite_repetitions)

        for n, tps in enumerate(tps_list):
            worksheet.update_cell(target_row, 16 + n, tps)
        
        worksheet.update_cell(target_row, 25, usage_vram)
